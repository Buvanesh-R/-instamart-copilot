"""Part D, tasks 5-8: tools, ReAct refund agent, safe stopping and crash-resume. Run this script for the full report.

Reads  data/instamart.db (Part A), data/raw/agent_scenarios.csv, retrieval from d1_rag.py
Writes outputs/agent_runs.jsonl (one line per scenario run), outputs/agent_demo_runs.jsonl (the stop-safely and
       crash-resume demonstrations), outputs/refund_ledger.jsonl and state/<run_id>.json checkpoints.
"""
import json
import os
import sqlite3
import time
from datetime import datetime, timezone

import pandas as pd

from common import DB_PATH, OUT, RAW, STATE, check, start_report
import mock_llm
from d1_rag import retrieve

LEDGER = os.path.join(OUT, "refund_ledger.jsonl")
RUNS = os.path.join(OUT, "agent_runs.jsonl")
DEMO_RUNS = os.path.join(OUT, "agent_demo_runs.jsonl")
_CTX = {"run_id": None}          # lets ledger lines say which run wrote them

AGENT_SYSTEM_PROMPT = ("You are the refund AGENT for Instamart support. Look up the order, check the refund policy, "
                       "calculate the refund and issue it. Escalate any refund above INR 1000 to a human. "
                       "Finish with a final answer that starts with REFUND, ESCALATE, DENIED or NOT_FOUND.")


# ----------------------------------------------------------------------------- task 5: tools
def get_order(order_id):
    con = sqlite3.connect(DB_PATH)
    try:
        row = con.execute("SELECT order_id, status, order_amount, delivery_minutes FROM orders WHERE order_id = ?",
                          (order_id,)).fetchone()
    finally:
        con.close()
    if row is None:
        return {"error": "order not found"}
    dm = row[3]
    return {"order_id": row[0], "status": row[1], "order_amount": row[2],
            "delivery_minutes": None if dm is None or dm != dm else dm}       # NULL / NaN -> null


def search_policy(query):
    return {"results": [{"source": h["source"], "score": round(h["score"], 4), "text": h["text"]}
                        for h in retrieve(query, k=3)]}


def calculate_refund(order_amount, delivery_minutes):
    """Refund rules: more than 45 minutes -> 100% of the order amount; 45 minutes or less -> 0."""
    if delivery_minutes is not None and delivery_minutes > 45:
        return {"refund_inr": round(float(order_amount), 2)}
    return {"refund_inr": 0}


def _ledger(tool, order_id, refund_inr):
    line = {"timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"), "tool": tool,
            "order_id": order_id, "refund_inr": refund_inr, "run_id": _CTX["run_id"]}
    with open(LEDGER, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(line) + "\n")
    return {"refund_inr": refund_inr}


def issue_refund(order_id, refund_inr):
    return _ledger("issue_refund", order_id, refund_inr)


def escalate_to_human(order_id, refund_inr):
    return _ledger("escalate_to_human", order_id, refund_inr)


def _spec(name, description, props):
    return {"type": "function", "function": {"name": name, "description": description,
            "parameters": {"type": "object", "properties": props, "required": list(props)}}}


TOOLS_SPEC = [
    _spec("get_order", "Read one order (status, amount, delivery minutes) from the database by order id.",
          {"order_id": {"type": "string", "description": "Order id such as ORD-100123"}}),
    _spec("search_policy", "Search the support policy documents and return the three best matching chunks.",
          {"query": {"type": "string", "description": "Free-text search query"}}),
    _spec("calculate_refund", "Apply the refund rules and return the refund amount in INR.",
          {"order_amount": {"type": "number"}, "delivery_minutes": {"type": "number"}}),
    _spec("issue_refund", "Issue an automatic refund of at most INR 1000 (appends to the refund ledger).",
          {"order_id": {"type": "string"}, "refund_inr": {"type": "number"}}),
    _spec("escalate_to_human", "Send a refund above INR 1000 to a human support lead (appends to the ledger).",
          {"order_id": {"type": "string"}, "refund_inr": {"type": "number"}}),
]
TOOL_MAP = {"get_order": get_order, "search_policy": search_policy, "calculate_refund": calculate_refund,
            "issue_refund": issue_refund, "escalate_to_human": escalate_to_human}


# ----------------------------------------------------------------------------- tasks 6-8: the agent loop
class SimulatedCrash(Exception):
    """Raised on purpose by the crash demonstration right after a checkpoint was saved."""


def _checkpoint_path(run_id):
    return os.path.join(STATE, f"{run_id}.json")


def _save(st):
    tmp = _checkpoint_path(st["run_id"]) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(st, fh, indent=1)
    os.replace(tmp, _checkpoint_path(st["run_id"]))       # atomic: a crash never leaves a half-written checkpoint


def _result(st, status, final_answer):
    refund = 0
    for s in st["steps"]:
        if s["tool"] in ("issue_refund", "escalate_to_human"):
            refund = s["observation"].get("refund_inr", 0)
    return {"run_id": st["run_id"], "scenario_id": st["scenario_id"], "status": status, "final_answer": final_answer,
            "steps": st["steps"], "llm_calls": st["llm_calls"], "tokens_used": st["tokens_used"], "refund_inr": refund,
            "tools_executed": list(st["tools_executed"])}


def _fallback(st, status, why):
    done = ", ".join(st["tools_executed"]) or "none"
    summary = (f"HANDOFF TO HUMAN: automated run stopped ({why}). Completed {len(st['steps'])} tool step(s) "
               f"[{done}] using {st['llm_calls']} model call(s) and {st['tokens_used']} tokens. "
               f"Customer message: {st['user_message']!r}")
    return _result(st, status, summary)


def _loop(st, crash_after_tools=None):
    """Run (or continue) the ReAct loop described by the state dict `st`."""
    _CTX["run_id"] = st["run_id"]
    t0 = time.monotonic()
    while st["llm_calls"] < st["max_iterations"]:
        elapsed = time.monotonic() - t0
        if elapsed >= st["time_limit_seconds"]:                                  # (c) wall-clock limit, top of iteration
            return _fallback(st, "FALLBACK_TIMEOUT", f"time limit of {st['time_limit_seconds']} s reached")
        reply = mock_llm.chat(st["messages"], tools=TOOLS_SPEC)
        st["llm_calls"] += 1
        st["tokens_used"] += reply.usage.total_tokens
        if st["tokens_used"] > st["token_limit"]:                                # (b) token limit, tool NOT executed
            return _fallback(st, "FALLBACK_TOKEN_LIMIT",
                             f"token limit {st['token_limit']} exceeded with {st['tokens_used']} tokens")
        msg = reply.choices[0].message
        if not msg.tool_calls:                                                   # final answer
            answer = msg.content or ""
            status = answer.split(":", 1)[0].strip() if ":" in answer else "UNKNOWN"
            return _result(st, status, answer)
        calls = [{"id": tc.id, "name": tc.name, "arguments": tc.arguments} for tc in msg.tool_calls]
        st["messages"].append({"role": "assistant", "content": None, "tool_calls": calls})
        for tc in calls:
            fn = TOOL_MAP.get(tc["name"])
            try:
                obs = fn(**tc["arguments"]) if fn else {"error": f"unknown tool {tc['name']}"}
            except Exception as exc:                                             # a failing tool becomes an observation
                obs = {"error": f"{type(exc).__name__}: {exc}"}
            st["messages"].append({"role": "tool", "tool_call_id": tc["id"], "name": tc["name"],
                                   "content": json.dumps(obs)})
            st["steps"].append({"step": len(st["steps"]) + 1, "tool": tc["name"], "arguments": tc["arguments"],
                                "observation": obs})
            st["tools_executed"].append(tc["name"])
            st["step"] = len(st["steps"])
            _save(st)                                                            # checkpoint after every observation
            if crash_after_tools is not None and len(st["tools_executed"]) == crash_after_tools:
                raise SimulatedCrash(f"simulated crash after tool execution #{crash_after_tools}")
    return _fallback(st, "FALLBACK_MAX_ITERATIONS", f"{st['max_iterations']} model calls without a final answer")


def run_agent(user_message, scenario_id, max_iterations=6, token_limit=6000, time_limit_seconds=10,
              run_id=None, crash_after_tools=None):
    st = {"run_id": run_id or f"run_{scenario_id}", "scenario_id": scenario_id, "user_message": user_message,
          "max_iterations": max_iterations, "token_limit": token_limit, "time_limit_seconds": time_limit_seconds,
          "messages": [{"role": "system", "content": AGENT_SYSTEM_PROMPT}, {"role": "user", "content": user_message}],
          "tokens_used": 0, "llm_calls": 0, "step": 0, "steps": [], "tools_executed": []}
    return _loop(st, crash_after_tools)


def resume_agent(run_id):
    """Load state/<run_id>.json and carry on from the saved messages; finished tools are never re-run."""
    with open(_checkpoint_path(run_id), encoding="utf-8") as fh:
        st = json.load(fh)
    return _loop(st)


def _line(res):
    return {k: res[k] for k in ("run_id", "scenario_id", "status", "final_answer", "llm_calls", "tokens_used",
                                "refund_inr", "steps")}


if __name__ == "__main__":
    start_report("d2_agent")
    for path in (LEDGER, RUNS, DEMO_RUNS):                       # fresh outputs on every run => reproducible
        if os.path.exists(path):
            os.remove(path)
    for fn in os.listdir(STATE):
        os.remove(os.path.join(STATE, fn))

    print("=== TASK 5: tools ===")
    print("  get_order('ORD-999999') ->", get_order("ORD-999999"))
    print("  calculate_refund(500, 46) ->", calculate_refund(500, 46), "| calculate_refund(500, 45) ->", calculate_refund(500, 45))
    print("  TOOLS_SPEC names:", [t["function"]["name"] for t in TOOLS_SPEC])

    print("\n=== TASK 6: run all 7 scenarios (max_iterations=6, token_limit=6000, time_limit_seconds=10) ===")
    sc = pd.read_csv(os.path.join(RAW, "agent_scenarios.csv"), keep_default_na=False)
    table = []
    for _, r in sc.iterrows():
        res = run_agent(r["message"], r["scenario_id"])
        with open(RUNS, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(_line(res)) + "\n")
        exp = r["expected_outcome"]
        ok_status = res["status"].startswith("FALLBACK") if exp == "FALLBACK" else res["status"] == exp
        ok_amt = abs(res["refund_inr"] - float(r["expected_refund_inr"])) <= 0.01
        table.append((r["scenario_id"], exp, res["status"], res["refund_inr"], float(r["expected_refund_inr"]),
                      res["llm_calls"], res["tokens_used"], ok_status and ok_amt))
        print(f"  {r['scenario_id']}: expected {exp:<9} got {res['status']:<24} refund {res['refund_inr']:>8.2f} "
              f"(expected {float(r['expected_refund_inr']):.2f}) model calls {res['llm_calls']} tokens {res['tokens_used']} "
              f"{'MATCH' if ok_status and ok_amt else 'MISMATCH'}")
        print(f"      final answer: {res['final_answer']}")
        print(f"      tools: {res['tools_executed']}")
    assert all(t[-1] for t in table), "a scenario did not match its expected outcome"
    s7 = [t for t in table if t[0] == "S7"][0]
    assert s7[2] == "FALLBACK_MAX_ITERATIONS" and s7[6] <= 6000
    check("S1 refund", table[0][3], 128.25, tol=0.01)
    check("S2 refund", table[1][3], 135.00, tol=0.01)
    check("S3 escalated amount", table[2][3], 1999.00, tol=0.01)

    print("\n=== TASK 7: stop safely ===")
    s1_msg = sc[sc["scenario_id"] == "S1"]["message"].iloc[0]
    d1 = run_agent(s1_msg, "S1", token_limit=200, run_id="demo_token_limit_S1")
    print(f"  (b) S1 with token_limit=200 -> {d1['status']}; model calls {d1['llm_calls']}, tokens {d1['tokens_used']}, "
          f"tools executed {d1['tools_executed']}")
    print(f"      final_answer: {d1['final_answer']}")
    assert d1["status"] == "FALLBACK_TOKEN_LIMIT" and "issue_refund" not in d1["tools_executed"]
    d2 = run_agent(s1_msg, "S1", time_limit_seconds=0, run_id="demo_timeout_S1")
    print(f"  (c) S1 with time_limit_seconds=0 -> {d2['status']}; model calls {d2['llm_calls']}, tools {d2['tools_executed']}")
    print(f"      final_answer: {d2['final_answer']}")
    assert d2["status"] == "FALLBACK_TIMEOUT"
    print(f"  (a) S7 -> {s7[2]} after {s7[5]} model calls and {s7[6]} tokens (default token_limit 6000 not hit)")

    print("\n=== TASK 8: crash and resume ===")
    rid = "demo_crash_resume_S1"
    try:
        run_agent(s1_msg, "S1", run_id=rid, crash_after_tools=2)
        raise SystemExit("the simulated crash did not happen")
    except SimulatedCrash as exc:
        print(f"  crash: {exc}")
    cp = json.load(open(_checkpoint_path(rid), encoding="utf-8"))
    print(f"  checkpoint state/{rid}.json: step {cp['step']}, tools executed so far {cp['tools_executed']}, "
          f"{len(cp['messages'])} messages, tokens_used {cp['tokens_used']}")
    res = resume_agent(rid)
    print(f"  resume_agent -> status {res['status']}, refund INR {res['refund_inr']:.2f}, final answer: {res['final_answer']}")
    print(f"  tools executed across crash + resume, in order: {res['tools_executed']}")
    assert res["tools_executed"] == ["get_order", "search_policy", "calculate_refund", "issue_refund"]
    assert res["status"] == "REFUND" and abs(res["refund_inr"] - 128.25) < 0.01
    for d in (d1, d2, res):
        with open(DEMO_RUNS, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(_line(d)) + "\n")

    ledger = [json.loads(x) for x in open(LEDGER, encoding="utf-8")]
    print(f"\n  refund ledger lines: {len(ledger)}")
    for ln in ledger:
        print("   ", ln["tool"], ln["order_id"], ln["refund_inr"], ln["run_id"])
    print("  state/ files:", sorted(os.listdir(STATE)))

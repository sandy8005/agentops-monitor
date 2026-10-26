"""
Migration 0020 — a rules-decided controller action records WHY. Idempotent.

  agent_actions.fallback_code TEXT
        For decided_by = 'rules': the metadata-only reason the rules policy decided
        instead of the model (llm.fallback_code: llm_budget_spent,
        cost_limit_reached, llm_unavailable, ..., or rules_only / rules_controller
        when the goal chose rules). NULL for model/backend decisions and for legacy
        rows (reason unrecorded). scripts/smoke_live.py accepts a rules fallback only
        with an allowed recorded reason. Job steps record the same codes in
        retrieved_context (requirements_fallback_code, judge_fallback_code).
"""


def upgrade(cur):
    cur.execute("ALTER TABLE agent_actions ADD COLUMN IF NOT EXISTS fallback_code TEXT")
    cur.execute("ALTER TABLE agent_actions DROP CONSTRAINT IF EXISTS agent_actions_fallback_code_chk")
    cur.execute("""
        ALTER TABLE agent_actions ADD CONSTRAINT agent_actions_fallback_code_chk CHECK (
            fallback_code IS NULL
            OR (decided_by = 'rules' AND fallback_code ~ '^[a-z_]{1,40}$'))""")

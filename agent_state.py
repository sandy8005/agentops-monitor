"""
AgentState: the single object threaded through the autonomous loop.
The planner reads it to decide the next action; tools update it.
Cache fields live here so the planner can skip a tool when its result
already exists. Budget methods (can_spend/spend) let logged_llm_call enforce
the per-run LLM request cap at the true unit of quota consumption.
"""


class AgentState:
    def __init__(self, goal, resume_id=None, target_role=None, location=None,
                 work_mode=None, employment_type=None, evaluate=False):
        # --- goal / inputs ---
        self.goal = goal
        self.resume_id = resume_id
        self.target_role = target_role
        self.location = location
        self.work_mode = work_mode
        self.employment_type = employment_type
        self.evaluate = evaluate

        # --- accumulated results ---
        self.resume_text = None
        self.parsed_resume = None
        self.jobs = None
        self.current_job_index = 0
        self.job_results = []
        self.ranked = None
        self.ranking_done = False        # True once the rank step has run (even if empty)
        self.advice_done = False         # True once advice generation has run
        self.cancelled = False           # set True when a cancel request is detected
        self.failed_jobs = 0             # count of jobs that errored during processing

        # --- caches ---
        self.requirements_cache = {}

        # --- budget (enforced INSIDE logged_llm_call, per actual HTTP attempt) ---
        self.llm_calls_made = 0
        self.max_llm_calls = 30

        # --- control ---
        self.completed_actions = []
        self.done = False
        self.error = None

    def record_action(self, action):
        self.completed_actions.append(action)

    def has(self, attr):
        val = getattr(self, attr, None)
        return val is not None and val != []

    # --- budget protocol: logged_llm_call calls these per real request ---
    def can_spend(self):
        """True if another actual LLM request is within budget."""
        return self.llm_calls_made < self.max_llm_calls

    def spend(self):
        """Record one actual LLM HTTP request (called per attempt, incl. retries)."""
        self.llm_calls_made += 1

    def budget_exceeded(self):
        return not self.can_spend()
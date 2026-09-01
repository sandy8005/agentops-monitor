"""
AgentState: the single object threaded through the autonomous loop.
The planner reads it to decide the next action; tools update it.
Cache fields live here so the planner can skip a tool when its result
already exists — this is where autonomy and call-reduction meet.
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

        # --- caches (populated by tools, checked by planner to skip work) ---
        self.requirements_cache = {}     # job_id -> extracted requirements

        # --- budget (call-reduction safety, enforced in the loop) ---
        self.llm_calls_made = 0
        self.max_llm_calls = 30          # hard per-run ceiling (#20)

        # --- control ---
        self.completed_actions = []
        self.done = False
        self.error = None

    def record_action(self, action):
        self.completed_actions.append(action)

    def has(self, attr):
        val = getattr(self, attr, None)
        return val is not None and val != []

    def budget_exceeded(self):
        return self.llm_calls_made >= self.max_llm_calls
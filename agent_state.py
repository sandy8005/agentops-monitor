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
        # --- human-in-the-loop review (LangGraph interrupt) ---
        self.last_job_needs_review = False   # did the just-processed job get flagged?
        self.last_review_step_id = None      # step_id of that job (for the interrupt payload)
        self.last_review_info = None         # dict shown to the human at the interrupt
        self.human_decisions = {}            # step_id -> {"decision":..., "comment":...}

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


    # --- Serialization for LangGraph checkpointing ---
    # Every field must round-trip: from_dict(to_dict(s)) must reproduce s exactly.
    # If a field is added to __init__, it MUST be added here too, or state silently
    # drops across a checkpoint pause/resume.

    _FIELDS = (
        "last_job_needs_review", "last_review_step_id", "last_review_info",
        "human_decisions",
        "goal", "resume_id", "target_role", "location", "work_mode",
        "employment_type", "evaluate", "resume_text", "parsed_resume", "jobs",
        "current_job_index", "job_results", "ranked", "ranking_done",
        "advice_done", "cancelled", "failed_jobs", "requirements_cache",
        "llm_calls_made", "max_llm_calls", "completed_actions", "done", "error",
    )

    def to_dict(self):
        """Flat, JSON-serializable snapshot of the entire state."""
        return {f: getattr(self, f) for f in self._FIELDS}

    @classmethod
    def from_dict(cls, d):
        """Reconstruct an AgentState from a to_dict() snapshot."""
        obj = cls(goal=d["goal"], resume_id=d.get("resume_id"),
                  target_role=d.get("target_role"), location=d.get("location"),
                  work_mode=d.get("work_mode"), employment_type=d.get("employment_type"),
                  evaluate=d.get("evaluate", False))
        for f in cls._FIELDS:
            if f in d:
                setattr(obj, f, d[f])
        return obj

    def __eq__(self, other):
        """Equality by serialized state — used to prove the round-trip is lossless."""
        if not isinstance(other, AgentState):
            return NotImplemented
        return self.to_dict() == other.to_dict()
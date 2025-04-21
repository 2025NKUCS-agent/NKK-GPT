class BinaryTrajectoryComparison(AbstractActionSampler):
    def __init__(self, config: BinaryTrajectoryComparisonConfig, model: AbstractModel, tools: ToolHandler):
        super().__init__(model, tools)
        self.config = config

    def _format_trajectory(self, trajectory: Trajectory) -> str:
        steps = []
        for i, step in enumerate(trajectory):
            steps.append(f"Action {i}: {step['action']}\n Observation {i}: {step['observation']}")
        return "\n".join(steps)

    def format_messages(
        self,
        *,
        problem_statement: ProblemStatement,
        trajectory: Trajectory,
        thought1: str,
        action1: str,
        thought2: str,
        action2: str,
        use_cache_control: bool = False,
    ) -> list[dict]:
        system_message = self.config.system_template
        self._logger.debug(f"MODEL INPUT (system)\n{system_message}")
        ps_format_dict = {
            "problem_statement": problem_statement.get_problem_statement(),
            **problem_statement.get_extra_fields(),
        }
        user_message = Template(self.config.instance_template).render(
            **ps_format_dict,
            traj=self._format_trajectory(trajectory),
        )
        self._logger.debug(f"MODEL INPUT (instance)\n{user_message}")
        comparison_message = Template(self.config.comparison_template).render(
            thought1=thought1,
            action1=action1,
            thought2=thought2,
            action2=action2,
        )
        self._logger.debug(f"MODEL INPUT (comparison)\n{comparison_message}")
        cache_control_kwargs = {"cache_control": {"type": "ephemeral"}} if use_cache_control else {}
        return [
            {"role": "system", "content": system_message},
            {
                "role": "user",
                "content": [{"type": "text", "text": user_message, **cache_control_kwargs}],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": comparison_message,
                    }
                ],
            },
        ]

    def filter_duplicates(self, completions: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Filter out duplicate actions"""
        thoughts: list[str] = []
        actions: list[str] = []
        filtered_completions: list[dict[str, Any]] = []
        for pc in completions:
            thought, action = self._tools.parse_actions(pc)
            if action not in actions:
                thoughts.append(thought)
                actions.append(action)
                filtered_completions.append(pc)

        if len(filtered_completions) < len(completions):
            self._logger.debug("Filtering duplicates: %d -> %d", len(completions), len(filtered_completions))

        return filtered_completions

    def filter_parseable_completions(self, completions: list[dict[str, Any]]) -> list[dict[str, Any]]:
        filtered_completions = []
        for completion in completions:
            try:
                self._tools.parse_actions(completion)
            except FormatError:
                self._logger.warning("Could not parse completion %s, skipping.", completion)
                continue
            filtered_completions.append(completion)
        if len(filtered_completions) == 0:
            msg = "No completions could be parsed."
            raise FormatError(msg)
        return filtered_completions

    def contains_edits(self, completions: list[dict[str, Any]]) -> bool:
        keywords = ["edit", "str_replace_editor insert", "str_replace_editor str_replace"]
        for completion in completions:
            _, action = self._tools.parse_actions(completion)
            if any(action.startswith(keyword) for keyword in keywords):
                return True
        return False

    def get_completions(self, history: list[dict[str, Any]]) -> list[dict[str, Any]]:
        completions = self._model.query(history, n=self.config.min_n_samples)  # type: ignore
        completions = self.filter_parseable_completions(completions)
        completions = self.filter_duplicates(completions)
        if not completions:
            msg = "No completions could be parsed."
            raise FormatError(msg)
        if self.contains_edits(completions) and self.config.min_n_samples < self.config.max_n_samples:
            self._logger.debug("Edits were proposed, will sample more")
            new_completions = self._model.query(history, n=self.config.max_n_samples - self.config.min_n_samples)  # type: ignore
            completions = self.filter_duplicates(self.filter_parseable_completions(completions + new_completions))
        if len(completions) == 1:
            _, action = self._tools.parse_actions(completions[0])
            self._logger.warning("Only identical actions were proposed (action=%s)", action)
        return completions

    def get_action(
        self,
        *,
        problem_statement: ProblemStatement,
        trajectory: Trajectory,
        history: list[dict[str, Any]],
    ) -> ActionSamplerOutput:
        completions = self.get_completions(history)
        best_idx = 0
        comparison_log = []
        for i in range(1, len(completions)):
            thought1, action1 = self._tools.parse_actions(completions[best_idx])
            thought2, action2 = self._tools.parse_actions(completions[i])
            messages = self.format_messages(
                problem_statement=problem_statement,
                trajectory=trajectory,
                thought1=thought1,
                action1=action1,
                thought2=thought2,
                action2=action2,
                use_cache_control=len(completions) >= 3,
            )
            response = self._model.query(messages, temperature=self.config.comparison_temperature)["message"]  # type: ignore
            self._logger.info(f"RESPONSE: {response}")
            idx = self.interpret(response)
            comparison_log.append(
                {
                    "comparison_between": (best_idx, i),
                    "messages": messages,
                    "response": response,
                    "idx": idx,
                }
            )
            best_idx = i if idx == 1 else best_idx

        return ActionSamplerOutput(
            completion=completions[best_idx],
            extra_info={"comparison_log": comparison_log},
        )

    def interpret(self, response: str) -> Literal[0, 1]:
        """Interpret response from LM. Note: 1-based indexing"""
        last_line = response.strip().split("\n")[-1].strip()
        if "first" in last_line.lower():
            return 0
        elif "second" in last_line.lower():
            return 1
        self._logger.warning("Could not interpret response: %s, will choose first submission.", response)
        return 0

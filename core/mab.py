class Bandit4Arm(BanditTask):
    """
    A class to hold a multi-armed bandit task data.
    """

    def __init__(
        self,
        probs,
        choices,
        rewards,
        session_ids,
        starts=None,
        stops=None,
        datetime=None,
        metadata=None,
    ):
        super().__init__(
            probs=probs,
            choices=choices,
            rewards=rewards,
            session_ids=session_ids,
            starts=starts,
            stops=stops,
            datetime=datetime,
            metadata=metadata,
        )
        assert self.n_ports == 4, "Bandit4Arm requires exactly 4 arms"
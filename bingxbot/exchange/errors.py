class BingXError(Exception):
    """Transport-level failure talking to BingX."""


class BingXAPIError(BingXError):
    """BingX returned a non-zero business code."""

    def __init__(self, code: int, msg: str, path: str = ""):
        self.code = code
        self.msg = msg
        self.path = path
        super().__init__(f"BingX {path} -> code={code} msg={msg}")

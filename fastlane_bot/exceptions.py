class FailedToGetTokenDetailsException(Exception):
    """
    Exception caused when token details are unable to be fetched by the contract
    """

    def __init__(self, addr):
        self.message = f"[events.managers.exceptions] Failed to get token symbol and decimals for token address: {addr}"

    def __str__(self):
        return self.message


class ReadOnlyException(Exception):
    def __init__(self, filepath):
        self.filepath = filepath

    def __str__(self):
        return (
            f"tokens.csv does not exist at {self.filepath}. Please run the bot without the `read_only` flag to "
            f"create this file."
        )


class AsyncUpdateRetryException(Exception):
    """
    Exception raised when async_update_pools_from_contracts fails and needs to be retried.
    """

    pass


class BalancerInputTooLargeError(AssertionError):
    pass


class BalancerOutputTooLargeError(AssertionError):
    pass


class ExchangeNotSupportedError(AssertionError):
    pass


class ExchangeInfoNotFound(AssertionError):
    pass

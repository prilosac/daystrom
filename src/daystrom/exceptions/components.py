class InvalidComponentError(Exception):
    """Exception raised for invalid components in the system."""

    def __init__(self, component_name: str = "Unknown Component", message: str = "Component invalid - no reason given"):
        self.component_name = component_name
        self.message = f"{component_name}: {message}"
        super().__init__(self.message)

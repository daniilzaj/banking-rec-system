from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    LOG_LEVEL: str = "INFO"
    INPUT_DIR: str = "data_input"
    OUTPUT_DIR: str = "data_output"

    # This tells Pydantic to load variables from a .env file
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding='utf-8')

# Create a single, importable instance of the settings
settings = Settings()
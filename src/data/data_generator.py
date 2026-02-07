"""Data generator - imports from retail generator."""

# Import the retail data generator as the main generator
from .data_generator_retail import RetailDataGenerator, load_config_and_generate

# For backward compatibility
SyntheticDataGenerator = RetailDataGenerator

__all__ = ['RetailDataGenerator', 'SyntheticDataGenerator', 'load_config_and_generate']
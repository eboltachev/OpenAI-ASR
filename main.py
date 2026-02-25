import logging
import sys

from config import state
from processor import app

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)

for _name, _model in state.model_mapping.items():
    app.register_model(name=_name, model=_model)
    logger.info(f"Registered model: {_name}:{_model}")

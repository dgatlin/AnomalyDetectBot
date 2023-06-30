import logging

from container.model_package.anomaly_model.config.core import PACKAGE_ROOT, config

logging.getLogger(config.app_config.package_name).addHandler(logging.NullHandler())

# todo update
# with open(PACKAGE_ROOT / "VERSION") as version_file:
#    __version__ = version_file.read().strip()

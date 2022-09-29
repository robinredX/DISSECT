from main_config import config
if config["type"]=="bulk":
	import Bulk_simulator
else:
	import ST_simulator

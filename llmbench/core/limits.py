"""Hard limits the benchmark form and its API enforce.

Every cap here exists to keep one request from turning into an unbounded run:
a repetition re-runs the whole prompt set, a fat model list loads every model
in turn, and a long configuration table multiplies all of it.
"""

# A configuration file is a short document; anything larger is a mistake and
# should not be read into memory or parsed.
MAX_UPLOAD_BYTES = 256 * 1024

# A cross-model run loads every model in turn; a long list is a long run and
# most machines fit a handful of models anyway.
MAX_MODELS = 6

# Configurations compared side by side in one run.
MAX_CONFIGURATIONS = 12

# Distinct prompts a run may carry.
MAX_PROMPTS = 20

# A repetition re-runs the whole prompt set, so the cap keeps a fat-fingered
# value from turning one comparison into an afternoon of generation.
MAX_REPETITIONS = 10

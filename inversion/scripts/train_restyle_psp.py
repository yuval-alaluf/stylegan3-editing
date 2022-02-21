import json
import pprint
import sys

import dataclasses
import pyrallis

sys.path.append(".")
sys.path.append("..")

from inversion.options.train_options import TrainOptions
from inversion.training.coach_restyle_psp import Coach


@pyrallis.wrap()
def main(opts: TrainOptions):
	opts.exp_dir.mkdir(exist_ok=True, parents=True)

	opts_dict = dataclasses.asdict(opts)
	pprint.pprint(opts_dict)
	with open(opts.exp_dir / 'opt.json', 'w') as f:
		json.dump(opts_dict, f, indent=4, sort_keys=True, default=str)

	coach = Coach(opts)
	coach.train()


if __name__ == '__main__':
	main()

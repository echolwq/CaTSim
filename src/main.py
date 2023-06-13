"""CaTSim runner."""

from utils import tab_printer
from CaTSim import CaTSimTrainer
from param_parser import parameter_parser

def main():
    """
    Parsing command line parameters, reading data.
    Fitting and scoring a CaTSim model.
    """
    args = parameter_parser()
    
    trainer = CaTSimTrainer(args)
    if args.load:
        trainer.load()
        tab_printer(args)
    else:
        tab_printer(args)
        trainer.fit()
                
    trainer.score()

if __name__ == "__main__":
    main()

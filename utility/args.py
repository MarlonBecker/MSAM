import configparser
import argparse as ar
from typing import Any


class MetaClass(type):
    def __getattr__(cls, key: str) -> Any:
        return cls.get(key)

class Args(metaclass=MetaClass):
    """
    Argument parsing used from default argpase module. Changes:
        - defaults are not allowed in source code
        - a default value must be provided for each argument in the input file
        - bool args are handeled as follows:
            -> --arg --> True
            -> --arg True --> True
            -> --arg False --> False
            -> (not given) --> default from ini file
    """
    data: ar.Namespace = None
    parser = ar.ArgumentParser()
    
    @classmethod
    def get(cls, key: str) -> Any:
        if cls.data is None:
            raise RuntimeError(f"Arguments were not parsed yet.")
        return getattr(cls.data, key)
    
    @classmethod
    def add_argument(cls, *args, **kwargs) -> None:
        if "default" in kwargs:
            raise RuntimeError("Defining defaults in source code is prohibited. Use input file instead.")
        if "type" in kwargs:
            if kwargs["type"] is bool:
                kwargs["nargs"] = "?"
                kwargs["const"] = True
        cls.parser.add_argument(*args, **kwargs)
    
    @classmethod
    def parse_args(cls) -> None:
        #only read input file name
        init_file_name_parser = ar.ArgumentParser(description="Blank Project",add_help=False)
        init_file_name_parser.add_argument("-i", "--ifile", help="input parameter file", default='configs/default.ini',metavar="FILE")
        args, remaining_argv = init_file_name_parser.parse_known_args()
        
        config = configparser.ConfigParser()
        config.optionxform = str # make config parse case sensitive
        out = config.read([args.ifile])
        if not out:
            raise RuntimeError(f"Input file '{args.ifile}' not found.")

        defaults = {}
        for sec in config.sections():
            defaults.update(**dict(config[sec]))

        default_keys = list(defaults.keys())
        for action in cls.parser._actions:
            if not isinstance(action, ar._HelpAction):
                if action.dest not in defaults:
                    raise RuntimeError(f"No default argument given for {action.dest}.")
                # necessary to convert strings to bool
                if action.type is bool:
                    action.type = make_bool
                # split lists seperated by whitespace in defaults file
                if action.nargs == "*":
                    defaults[action.dest] = defaults[action.dest].split(" ") if defaults[action.dest] else []

                default_keys.remove(action.dest)

        if default_keys:
            raise RuntimeError(f"Parameter(s) {default_keys} were given in input file but not defined before.")

        cls.parser.set_defaults(**defaults)
        cls.data = cls.parser.parse_args(remaining_argv)
        
    @classmethod
    def parse_args_contin(cls, defaults: dict) -> None:
        defaults["contin"] = True #make sure contin argument is not overwritten by defaults from param.json file
        cls.parser.set_defaults(**defaults) #read defaults from param.json file
        
        # parse ifile argument here only to not throw an error by real parser
        init_file_name_parser = ar.ArgumentParser(description="",add_help=False)
        init_file_name_parser.add_argument("-i", "--ifile", help="input parameter file", default='input.ini',metavar="FILE")
        _, remaining_argv = init_file_name_parser.parse_known_args()
        
        cls.data = cls.parser.parse_args(remaining_argv)



def make_bool(arg: Any) -> bool:
    if isinstance(arg, str):
        if arg.lower() in ['true', '1', 't', 'y', 'yes']:
            return True
        elif arg.lower() in ['false', '0', 'f', 'n', 'no']:
            return False
        else:
            raise RuntimeError(f"Could not convert string to bool: {arg}")
    else:
        return bool(arg)

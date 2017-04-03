import argparse
import json
import logging
import os
import subprocess
from pathlib import Path

logger = logging.getLogger("Launch-many")

class Launcher:
    """
    Launcher class for calling dq to launch train.py with a list of 
    hyperparameter. 
    """
    def __init__(self, default_config, param, experiment_name):
        """
        :param default_config: the default config file
        :param param: the parameters to be overridden in the base config
                     (these are the hyper parameters we search on and output 
                      path)
        :param experiment_name: name of the experiment
        """
        
        output_dir = default_config.get("io").get("output_save_path")
        output_dir = os.path.join( output_dir, experiment_name)

        override_config = default_config
        logger.debug("override cfg starts as " + str(override_config))
        
        for val in param.get("value"):
            logger.debug("Hyper param value: " + str(val))
            
            #TODO not tested for more than one hyperparams
            for k, v in override_config.items():
                self.search_and_replace_dict(v, param.get("name"), val)
            
            output_cfg_path, output_dir_exp = self.get_cfg_path(output_dir, 
                    param.get("name"), val)
            override_config['io']['output_save_path'] = output_dir_exp
            
            if not os.path.exists(output_dir_exp):
                logger.debug("Creating " + str(output_dir_exp))
                path = Path(output_dir_exp)
                path.mkdir(parents=True)
            
            with open(output_cfg_path, 'w') as fp:
                json.dump(override_config, fp)

            #call dq TODO
            DQ_CFG = output_cfg_path
            os.system("DQ_CFG=%s dq-submit launch.sh"%output_cfg_path)
#p = subprocess.Popen(["dq-submit", "launch.sh"]
#"python", "train.py", "-c",
#                output_cfg_path], stdout=subprocess.PIPE)
            #p.communicate()


    def get_cfg_path(self, output_dir, param_name, val):
        dr = os.path.join(output_dir , param_name + "_" + str(val))
        cfg = os.path.join(dr, "override.cfg")
        logger.debug("dir: " + str(dr) + " cfg: " + str(cfg))
        return cfg, dr
   

    #TODO: refactor later
    def search_and_replace_dict(self, cfg, param, val):

        if isinstance(cfg, dict):
            for k, v in cfg.items():
                
                # is a dict itself
                if isinstance(v, dict):
                    return self.search_and_replace_dict(v, param, val)
                
                # is not a dict, so search with self
                else:
                    # if not a dict, then look in current
                    if k == param:
                        cfg[k] = val
                        return
        
        return



def get_default_config():
    retVal = {
	"seed" : 4337,
	"data" : { "path" : "/deep/group/med/alivecor/training2017", "seed" : 2016 },
	"optimizer" : { "name": "momentum", "epochs" : 50, "learning_rate" : 1e-2,
	    "momentum" : 0.95, "decay_rate" : 1.0, "decay_steps" : 2000
	},
	"model" : { "dropout" : 0.5, "batch_size" : 32,
	    "conv_layers" : [
		{ "filter_size" : 256, "num_filters" : 64, "stride" : 7 },
		{ "filter_size" : 128, "num_filters" : 64, "stride" : 7 }
	   ] },
	"io" : {
	    "output_save_path" : "/tmp"
	}
    }
    return retVal


def main():
    parser = argparse.ArgumentParser(description="Data Loader")
    parser.add_argument("-v", "--verbose", default=False, action="store_true")
    parser.add_argument("-e", "--experiment_name", default=None)
    parser.add_argument("-b", "--base_config", default=None)
    parser.add_argument("-p", "--parameter_config", default=None)

    parsed_arguments = parser.parse_args()
    arguments = vars(parsed_arguments)

    is_verbose       = arguments['verbose']
    base_config      = arguments['base_config']
    param_config     = arguments['parameter_config']
    experiment_name  = arguments['experiment_name'].rstrip()

    if is_verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)
    

    default_config = {}
    param          = {}
    if not base_config:
        default_config = get_default_config()
        logger.debug("No base cfg supplied, so using default - " + str(base_config))
    else:
        with open(base_config) as fid:
            default_config = json.load(fid)
    
    if not param_config:
        raise ValueError("Mandatory to specify parameter config")
    else:
        with open(param_config) as fid:
            param = json.load(fid)
    
    if not experiment_name:
        raise ValueError("Mandatory to specify experiment name")

    launcher = Launcher(default_config, param, experiment_name)



if __name__ == '__main__':
    main()


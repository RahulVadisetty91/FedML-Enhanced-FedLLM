import os
import platform
import shutil
from os.path import expanduser

import click
from fedml.core.common.singleton import Singleton
from fedml.cli.comm_utils.yaml_utils import load_yaml_config
from fedml.cli.cli_utils import platform_is_valid
from constants import Constants
from app_manager import FedMLAppManager
from job_manager import FedMLJobManager


class FedMLLaunchManager(Singleton):
    def __init__(self):
        pass

    @staticmethod
    def get_instance():
        return FedMLLaunchManager()

    def set_config_version(self, config_version):
        if config_version is not None:
            self.config_version = config_version

    def launch_job(self, yaml_file, user_id, user_api_key):
        if os.path.dirname(yaml_file) == "":
            yaml_file = os.path.join(os.getcwd(), yaml_file)

        self.parse_job_yaml(yaml_file)

        # Generate source, config and bootstrap related paths.
        platform_str = Constants.FEDML_PLATFORM_FALCON_STR
        platform_type = Constants.FEDML_PLATFORM_FALCON_TYPE
        client_server_type = Constants.FEDML_PACKAGE_BUILD_TARGET_TYPE_CLIENT
        source_full_path = os.path.join(self.job_config.base_dir, self.job_config.executable_file)
        source_full_folder = os.path.dirname(source_full_path)
        source_folder = os.path.dirname(self.job_config.executable_file)
        entry_point = os.path.basename(self.job_config.executable_file)
        config_full_path = os.path.join(self.job_config.base_dir, self.job_config.executable_conf)
        config_full_folder = os.path.dirname(config_full_path)
        config_folder = os.path.dirname(self.job_config.executable_conf)
        dest_folder = os.path.join(Constants.get_fedml_home_dir(), Constants.FEDML_LAUNCH_JOB_TEMP_DIR)
        bootstrap_full_path = os.path.join(config_full_folder, Constants.BOOTSTRAP_FILE_NAME)
        bootstrap_file = os.path.join(config_folder, Constants.BOOTSTRAP_FILE_NAME)
        if platform.system() == Constants.OS_PLATFORM_WINDOWS:
            bootstrap_full_path = bootstrap_full_path.replace('.sh', '.bat')
        os.makedirs(dest_folder, exist_ok=True)

        # Check the paths.
        if not os.path.exists(source_full_path):
            click.echo(f"Executable file {source_full_path} does not exist. Please check it.")
            return None
        if not os.path.exists(config_full_path):
            os.makedirs(config_folder, exist_ok=True)
            with open(config_full_path, 'w') as config_file_handle:
                config_file_handle.writelines(["environment_args:", f"  bootstrap: {bootstrap_file}"])
                config_file_handle.close()

        # Write pre setup commands into the bootstrap file.
        configs = load_yaml_config(config_full_path)
        configs[Constants.STD_CONFIG_ENV_SECTION][Constants.STD_CONFIG_ENV_SECTION_BOOTSTRAP_KEY] = bootstrap_file
        Constants.generate_yaml_doc(configs, config_full_path)
        with open(bootstrap_full_path, 'w') as bootstrap_file_handle:
            bootstrap_file_handle.writelines(self.job_config.pre_setup)
            bootstrap_file_handle.writelines(self.job_config.run_commands)
            bootstrap_file_handle.close()

        build_result_package = FedMLLaunchManager.build_job_package(platform_str, client_server_type, source_full_folder,
                                                                    entry_point, config_full_folder, dest_folder, "")
        if build_result_package is None:
            click.echo("Failed to build the application package for the executable file.")
            return None

        FedMLAppManager.get_instance().set_config_version(self.config_version)
        app_updated_result = FedMLAppManager.get_instance().update_app(platform_type, self.job_config.application_name,
                                                                       user_id, user_api_key,
                                                                       client_package_file=build_result_package)
        if not app_updated_result:
            click.echo("Failed to upload the application package to MLOps.")
            return None

        launch_result = FedMLJobManager.get_instance().start_job(platform_str, self.job_config.project_name,
                                                                 self.job_config.application_name,
                                                                 "[]", user_id, user_api_key)
        launch_result.project_name = self.job_config.project_name
        return launch_result

    def parse_job_yaml(self, yaml_file):
        self.job_config = FedMLJobConfig(yaml_file)

    @staticmethod
    def build_job_package(platform, client_server_type, source_folder, entry_point,
                          config_folder, dest_folder, ignore, verbose=False):
        if verbose:
            print("Argument for type: " + client_server_type)
            print("Argument for source folder: " + source_folder)
            print("Argument for entry point: " + entry_point)
            print("Argument for config folder: " + config_folder)
            print("Argument for destination package folder: " + dest_folder)
            print("Argument for ignore lists: " + ignore)

        if not platform_is_valid(platform):
            return

        if client_server_type == "client" or client_server_type == "server":
            if verbose:
                print(
                    "Now, you are building the fedml packages which will be used in the MLOps "
                    "platform."
                )
                print(
                    "The packages will be used for client training and server aggregation."
                )
                print(
                    "When the building process is completed, you will find the packages in the directory as follows: "
                    + os.path.join(dest_folder, "dist-packages")
                    + "."
                )
                print(
                    "Then you may upload the packages on the configuration page in the MLOps platform to start the "
                    "federated learning flow."
                )
                print("Building...")
        else:
            if verbose:
                print("You should specify the type argument value as client or server.")
            exit(-1)

        home_dir = expanduser("~")
        mlops_build_path = os.path.join(home_dir, "fedml-mlops-build")
        try:
            shutil.rmtree(mlops_build_path, ignore_errors=True)
        except Exception as e:
            pass

        ignore_list = "{},{}".format(ignore, Constants.FEDML_MLOPS_BUILD_PRE_IGNORE_LIST)
        pip_source_dir = os.path.dirname(__file__)
        pip_build_path = os.path.join(pip_source_dir, "build-package")
        build_dir_ignore = "__pycache__,*.pyc,*.git"
        build_dir_ignore_list = tuple(build_dir_ignore.split(','))
        shutil.copytree(pip_build_path, mlops_build_path,
                        ignore_dangling_symlinks=True, ignore=shutil.ignore_patterns(*build_dir_ignore_list))

        if client_server_type == "client":
            result = FedMLLaunchManager.build_mlops_package(
                ignore_list,
                source_folder,
                entry_point,
                config_folder,
                dest_folder,
                mlops_build_path,
                "fedml-client",
                "client-package",
                "${FEDSYS.CLIENT_INDEX}",
            )
            if result != 0:
                exit(result)

            build_result_package = os.path.join(dest_folder, "dist-packages", "client-package.zip")
            if verbose:
                print("You have finished all building process. ")
                print(
                    "Now you may use "
                    + build_result_package
                    + " to start your federated "
                      "learning run."
                )

            return build_result_package
        elif client_server_type == "server":
            result = FedMLLaunchManager.build_mlops_package(
                ignore_list,
                source_folder,
                entry_point,
                config_folder,
                dest_folder,
                mlops_build_path,
                "fedml-server",
                "server-package",
                "0",
            )
            if result != 0:
                exit(result)

            build_result_package = os.path.join(dest_folder, "dist-packages", "server-package.zip")
            if verbose:
                print("You have finished all building process. ")
                print(
                    "Now you may use "
                    + os.path.join(dest_folder, "dist-packages", "server-package.zip")
                    + " to start your federated "
                      "learning run."
                )

            return build_result_package

    @staticmethod
    def build_mlops_package(
            ignore,
            source_folder,
            entry_point,
            config_folder,
            dest_folder,
            mlops_build_path,
            mlops_package_parent_dir,
            mlops_package_name,
            rank,
            verbose=False
    ):
        if not os.path.exists(source_folder):
            if verbose:
                print("source folder is not exist: " + source_folder)
            return -1

        if not os.path.exists(os.path.join(source_folder, entry_point)):
            if verbose:
                print(
                    "entry file: "
                    + entry_point
                    + " is not exist in the source folder: "
                    + source_folder
                )
            return -1

        if not os.path.exists(config_folder):
            if verbose:
                print("config folder is not exist: " + source_folder)
            return -1

        mlops_src = source_folder
        mlops_src_entry = entry_point
        mlops_conf = config_folder
        cur_dir = mlops_build_path
        mlops_package_base_dir = os.path.join(
            cur_dir, "mlops-core", mlops_package_parent_dir
        )
        package_dir = os.path.join(mlops_package_base_dir, mlops_package_name)
        fedml_dir = os.path.join(package_dir, "fedml")
        mlops_dest = fedml_dir
        mlops_dest_conf = os.path.join(fedml_dir, "config")
        mlops_pkg_conf = os.path.join(package_dir, "conf", "fedml.yaml")
        mlops_dest_entry = os.path.join("fedml", mlops_src_entry)
        mlops_package_file_name = mlops_package_name + ".zip"
        dist_package_dir = os.path.join(dest_folder, "dist-packages")
        dist_package_file = os.path.join(dist_package_dir, mlops_package_file_name)
        ignore_list = tuple(ignore.split(','))

        shutil.rmtree(mlops_dest_conf, ignore_errors=True)
        shutil.rmtree(mlops_dest, ignore_errors=True)
        try:
            shutil.copytree(mlops_src, mlops_dest, copy_function=shutil.copy,
                            ignore_dangling_symlinks=True, ignore=shutil.ignore_patterns(*ignore_list))
        except Exception as e:
            pass
        try:
            shutil.copytree(mlops_conf, mlops_dest_conf, copy_function=shutil.copy,
                            ignore_dangling_symlinks=True, ignore=shutil.ignore_patterns(*ignore_list))
        except Exception as e:
            pass
        try:
            os.remove(os.path.join(mlops_dest_conf, "mqtt_config.yaml"))
            os.remove(os.path.join(mlops_dest_conf, "s3_config.yaml"))
        except Exception as e:
            pass

        mlops_pkg_conf_file = open(mlops_pkg_conf, mode="w")
        mlops_pkg_conf_file.writelines(
            [
                "entry_config: \n",
                "  entry_file: " + mlops_dest_entry + "\n",
                "  conf_file: config/fedml_config.yaml\n",
                "dynamic_args:\n",
                "  rank: " + rank + "\n",
                "  run_id: ${FEDSYS.RUN_ID}\n",
                # "  data_cache_dir: ${FEDSYS.PRIVATE_LOCAL_DATA}\n",
                # "  data_cache_dir: /fedml/fedml-package/fedml/data\n",
                "  mqtt_config_path: /fedml/fedml_config/mqtt_config.yaml\n",
                "  s3_config_path: /fedml/fedml_config/s3_config.yaml\n",
                "  log_file_dir: /fedml/fedml-package/fedml/data\n",
                "  log_server_url: ${FEDSYS.LOG_SERVER_URL}\n",
                "  client_id_list: ${FEDSYS.CLIENT_ID_LIST}\n",
                "  client_objects: ${FEDSYS.CLIENT_OBJECT_LIST}\n",
                "  is_using_local_data: ${FEDSYS.IS_USING_LOCAL_DATA}\n",
                "  synthetic_data_url: ${FEDSYS.SYNTHETIC_DATA_URL}\n",
                "  client_num_in_total: ${FEDSYS.CLIENT_NUM}\n",
            ]
        )
        mlops_pkg_conf_file.flush()
        mlops_pkg_conf_file.close()

        local_mlops_package = os.path.join(mlops_package_base_dir, mlops_package_file_name)
        if os.path.exists(local_mlops_package):
            os.remove(os.path.join(mlops_package_base_dir, mlops_package_file_name))
        mlops_archive_name = os.path.join(mlops_package_base_dir, mlops_package_name)
        shutil.make_archive(
            mlops_archive_name,
            "zip",
            root_dir=mlops_package_base_dir,
            base_dir=mlops_package_name,
        )
        if not os.path.exists(dist_package_dir):
            os.makedirs(dist_package_dir, exist_ok=True)
        if os.path.exists(dist_package_file) and not os.path.isdir(dist_package_file):
            os.remove(dist_package_file)
        mlops_archive_zip_file = mlops_archive_name + ".zip"
        if os.path.exists(mlops_archive_zip_file):
            shutil.move(mlops_archive_zip_file, dist_package_file)

        shutil.rmtree(mlops_build_path, ignore_errors=True)

        return 0


'''
Job yaml file is as follows:
fedml_params:
    fedml_account_id: 1111
    project_name: customer_service_llm
    job_name: fine_day
development_resources:
    dev_env: tbd        # development resources bundle to load on each machine
    network: tbd        # network protocol for communication between machines
executable_code_and_data:
    executable_file: tbd       # your main executable file, which can be empty
    executable_conf: tbd       # your config file for the main executable program, which can be empty
    data_location: tbd         # path to your data
    # pre setup shell commands. support multiple lines, which can be empty
    pre_setup:  | 
        tbd  
        tbd       
    # run shell commands. support multiple lines, which can be empty, but at least one of run_commands and excecutable_file must not be empty. 
    # Both the run_commands and executable_file can be the main executable commands. First run run_commands, then run executable_file. 
    run_commands: | 
        tbd  
        tbd      
gpu_requirements:
    minimum_num_gpus: 8             # minimum # of GPUs to provision
    maximum_cost_per_hour: $1.75    # max cost per hour for your job per machine
'''


class FedMLJobConfig(object):
    def __init__(self, job_yaml_file):
        job_config = load_yaml_config(job_yaml_file)
        self.account_id = job_config["fedml_params"]["fedml_account_id"]
        self.project_name = job_config["fedml_params"]["project_name"]
        self.job_name = job_config["fedml_params"]["job_name"]
        self.dev_env = job_config["development_resources"]["dev_env"]
        self.network = job_config["development_resources"]["network"]
        self.base_dir = os.path.dirname(job_yaml_file)
        self.executable_file = job_config["executable_code_and_data"]["executable_file"]
        self.executable_conf = job_config["executable_code_and_data"]["executable_conf"]
        self.executable_file = str(self.executable_file).replace('\\', os.sep).replace('/', os.sep)
        self.executable_conf = str(self.executable_conf).replace('\\', os.sep).replace('/', os.sep)
        self.data_location = job_config["executable_code_and_data"]["data_location"]
        self.pre_setup = job_config["executable_code_and_data"]["pre_setup"]
        self.run_commands = job_config["executable_code_and_data"]["run_commands"]
        self.minimum_num_gpus = job_config["gpu_requirements"]["minimum_num_gpus"]
        self.maximum_cost_per_hour = job_config["gpu_requirements"]["maximum_cost_per_hour"]
        self.application_name = f"App-{self.job_name}"

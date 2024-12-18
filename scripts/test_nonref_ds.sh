script_dir=$(dirname $0)
proj_dir=$(dirname ${script_dir})
source ${script_dir}/ansi_escape.sh


################## parse arguments and options ###############################
short_opt=h,t
long_opt=help,dry_run,load_prefix:
options=$(getopt -a --options ${short_opt} --longoptions ${long_opt} -- "$@")
# echo ${options}

eval set -- "${options}"

help_str="
Usage:
 $0 model_name net_cfg name epochs [options]

Test model.

Positional Arguments:
 model_name           model name
 net_cfg              path to network config file
 name                 name of training progress
 epochs               epochs

Options:
 --load_prefix <load_prefix>    prefix of the filename of weight file
 --ds_dict <ds_cfgs>            an associative array that records the dataset and its configuration
 -t, --dry_run                  just test not run actually
 -h, --help                     display this help
"

load_prefix=weights
ds_dict=${script_dir}/ds_dict/nonref_default.sh
dry_run=

while true; do
  case "$1" in
    --load_prefix)
      load_prefix="${2}"
      shift 2
      ;;
    --ds_dict)
      ds_dict="${2}"
      shift 2
      ;;
    -t | --dry_run)
      dry_run="echo "
      shift 1
      ;;
    -h | --help)
      echo -e "${help_str}"
      exit 0
      ;;
    --)
      shift
      break
      ;;
    *)
      echo "Unexpected option: $1"
      ;;
  esac
done

if [ $# -lt 4 ]
then
    echo "${help_str}"
    exit 1
fi

model_name=${1}
net_cfg=${2}
name=${3}
epochs=${4}
##############################################################################

source ${ds_dict}

for ds_name in ${!ds_dict[@]};
do
    ${dry_run}python ${proj_dir}/test.py \
    --model_name ${model_name} \
    --ds_cfg ${ds_dict[${ds_name}]} \
    --net_cfg ${net_cfg} \
    --name ${name} \
    --test_name ${ds_name} \
    --epochs ${epochs} \
    --load_prefix ${load_prefix}
done
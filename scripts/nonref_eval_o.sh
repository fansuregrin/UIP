script_dir=$(dirname $0)
proj_dir=$(dirname ${script_dir})
source ${script_dir}/ansi_escape.sh

################## parse arguments and options ###############################
short_opt=h,t
long_opt=help,dry_run,sub_path:,ds_names:,output_format:
options=$(getopt -a --options ${short_opt} --longoptions ${long_opt} -- "$@")
# echo ${options}

eval set -- "${options}"

help_str="
Usage:
 $0 result_dir [options]

Non-reference Evaluation.

Positional Arguments:
 result_dir                     directory to image results

Options:
 --sub_path <sub_path>          relative path based on result_dir to images floder
 --ds_names <ds_names>          test dataset names
 --output_format <format>...    format for saving evaluation data
 -t, --dry_run                  just test not run actually
 -h, --help                     display this help
"

sub_path=
output_format="csv pkl"
ds_names="U45 RUIE_Color90 UPoor200 UW2023"
dry_run=

while true; do
  case "$1" in
    --sub_path)
      sub_path="$2"
      shift 2
      ;;
    --ds_names)
      ds_names="$2"
      shift 2
      ;;
    --output_format)
      output_format="$2"
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

if [ $# -lt 1 ];then
    echo -e "${help_str}"
    exit 1
fi

res_dir="${1}"
################## parse arguments and options ###############################


ds_names_arr=(${ds_names})
for ds_name in ${ds_names_arr[@]}
do
    target_dir="${res_dir}/${ds_name}"
    if [ -d ${target_dir} ]
    then
        ${dry_run}python ${proj_dir}/nonref_eval_pd.py \
            -inp "${target_dir}/${sub_path}" \
            -out "${target_dir}"
            --output_format ${output_format}
    else
        echo -e "${RED}[${target_dir}]${ENDSTYLE} not exist!"
    fi
done

${dry_run}python ${script_dir}/get_eval_o.py ${res_dir} \
    --eval_type nonref \
    --ds_names ${ds_names} \
    --metric_names uranker uciqe uiqm
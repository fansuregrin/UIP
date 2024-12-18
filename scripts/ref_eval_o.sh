script_dir=$(dirname $0)
proj_dir=$(dirname ${script_dir})
source ${script_dir}/ansi_escape.sh

################## parse arguments and options ###############################
short_opt=h,t
long_opt=help,subpath:,width:,height:,resize,output_format:,window_size:,dry_run
options=$(getopt -a --options ${short_opt} --longoptions ${long_opt} -- "$@")
# echo ${options}

eval set -- "${options}"

help_str="
Usage:
 $0 result_dir [options]

Reference-based Evaluation.

Positional Arguments:
 result_dir                     directory to image results

Options:
 --sub_path <sub_path>          relative path based on result_dir to images floder
 --width <width>                image width to resize
 --height <height>              image height to resize
 --resize                       resize image
 --output_format <format>...    format for saving evaluation data
 --window_size <window_size>    window size for calculating ssim
 -t, --dry_run                  just test not run actually
 -h, --help                     display this help
"

sub_path=
width=256
height=256
resize=
output_format="csv pkl"
window_size=11
dry_run=

while true; do
  case "$1" in
    --sub_path)
      sub_path="$2"
      shift 2
      ;;
    --width)
      width="$2"
      shift 2
      ;;
    --height)
      height="$2"
      shift 2
      ;;
    --resize)
      resize="--resize"
      shift 1
      ;;
    --output_format)
      output_format="$2"
      shift 2
      ;;
    --window_size)
      window_size="$2"
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

declare -A refer_dict
refer_dict=([EUVP515]="/DataA/pwz/workshop/Datasets/EUVP_Dataset/test_samples/GTr"
            [OceanEx]="/DataA/pwz/workshop/Datasets/ocean_ex/good"
            [UIEB100]="/DataA/pwz/workshop/Datasets/UIEB100/reference"
            [LSUI_Test]="/DataA/pwz/workshop/Datasets/LSUI/test/ref")

for ds_name in ${!refer_dict[@]}
do
    target_dir="${res_dir}/${ds_name}"
    if [ -d ${target_dir} ]
    then
        ${dry_run}python ${proj_dir}/ref_eval_pd.py \
            -inp "${target_dir}/${sub_path}" \
            -ref "${refer_dict[${ds_name}]}" \
            -out "${target_dir}" \
            ${resize} --width ${width} --height ${height} \
            --output_format ${output_format} \
            --window_size ${window_size}
    else
        echo -e "${RED}[${target_dir}]${ENDSTYLE} not exist!"
    fi
done

ds_names=$(echo "${!refer_dict[@]}")
${dry_run}python ${script_dir}/get_eval_o.py ${res_dir} \
    --eval_type ref \
    --ds_names ${ds_names} \
    --metric_names psnr ssim
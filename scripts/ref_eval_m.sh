script_dir=$(dirname $0)
proj_dir=$(dirname ${script_dir})
source ${script_dir}/ansi_escape.sh


################## parse arguments and options ###############################
short_opt=h,t
long_opt=help,dry_run,load_prefix:,width:,height:,resize,output_format:,window_size:,refer_dict:
options=$(getopt -a --options ${short_opt} --longoptions ${long_opt} -- "$@")
# echo ${options}

eval set -- "${options}"

help_str="
Usage:
 $0 model_name network name epochs [options]

Reference-based Evaluation.

Positional Arguments:
 model_name                     name of the model
 network                        name of the network
 name                           name of traing progress
 epochs                         epochs

Options:
 --load_prefix <load_prefix>    prefix of the filename of weight file                  
 --width <width>                image width to resize
 --height <height>              image height to resize
 --resize                       resize image
 --output_format <format>...    format for saving evaluation data
 --window_size <window_size>    window size for calculating ssim
 --refer_dict <refer_dict>      
 -t, --dry_run                  just test not run actually
 -h, --help                     display this help
"

load_prefix=weights
width=256
height=256
resize=
output_format="csv pkl"
window_size=11
refer_dict="${script_dir}/refer_dict/default.sh"
dry_run=

while true; do
  case "$1" in
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
    --refer_dict)
      refer_dict="$2"
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

if [ $# -lt 4 ];then
    echo -e "${help_str}"
    exit 1
fi

model_name=${1}
net=${2}
name=${3}
epochs=${4}
################## parse arguments and options ###############################


source ${refer_dict}
epoch_arr=(${epochs})
for ds_name in ${!refer_dict[@]}
do
    for epoch in ${epoch_arr[*]}
    do
        target_dir="${proj_dir}/results/${model_name}/${net}/${name}/${ds_name}/${load_prefix}_${epoch}"
        if [ -d ${target_dir} ]
        then
            ${dry_run}python ${proj_dir}/ref_eval_pd.py \
                -inp "${target_dir}/single/predicted" \
                -ref "${refer_dict[${ds_name}]}" \
                -out "${target_dir}" \
                ${resize} --width ${width} --height ${height} \
                --output_format ${output_format} \
                --window_size ${window_size}
        else
            echo -e "${RED}[${target_dir}]${ENDSTYLE} not exist!"
        fi
    done
done

ds_names=$(echo "${!refer_dict[@]}")
${dry_run}python ${script_dir}/get_eval_m.py \
    ${model_name} \
    ${net} \
    ${name} \
    ${epochs} \
    --eval_type ref \
    --ds_names ${ds_names} \
    --metric_names psnr ssim \
    --load_prefix ${load_prefix}
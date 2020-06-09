
evaluate_matching(){ # first arg = model save_name
  orig=$(pwd)
  base=$1
  id=$2

  cd $base/HardNet_MultiDataset
#  rm -rf HP_descs/* # run this from time to time
  python -utt HPatches_extract_HardNet.py --model_name=$id --overwrite --hpatches_dir="$base/hpatches-release"

  cd "$base/hpatches-benchmark/python"
  name=HardNet_$id
  dir="$base/HardNet_MultiDataset/HP_descs"
  python -utt hpatches_eval.py --descr-dir=$dir --descr-name=$name --task=matching --split=illum --del=1
  python -utt hpatches_eval.py --descr-dir=$dir --descr-name=$name --task=matching --split=view --del=1
  python -utt hpatches_eval.py --descr-dir=$dir --descr-name=$name --task=matching --split=full --del=1

  python -utt hpatches_results.py --results-dir=results/ --descr-name=$name --task=matching --split=illum
  python -utt hpatches_results.py --results-dir=results/ --descr-name=$name --task=matching --split=view
  python -utt hpatches_results.py --results-dir=results/ --descr-name=$name --task=matching --split=full

  cd $base/HardNet_MultiDataset
  p test.py match $id # also match on AMOS
  cd $orig
}

rootsift(){
  name=rootsift
  sep=';'
  python hpatches_eval.py --descr-dir=/home/milan/Prace/CMP/hardnet/benchmarks/out --descr-name=$name --task=verification --task=matching --task=retrieval --split=full --delimiter=sep
  python hpatches_results.py --results-dir=results/ --descr-name=$name --task=verification --task=matching --task=retrieval --split=full
}

verification(){ # --task=verification --task=matching --task=retrieval
  name=$1
  dir="../../hardnet/benchmarks/out"
  python hpatches_eval.py --descr-dir=$dir --descr-name=$name --task=verification --split=full --del=1
  python hpatches_results.py --results-dir=results/ --descr-name=$name --task=verification --split=full
}

eval_retrieval(){ # --task=verification --task=matching --task=retrieval
  orig=$(pwd)

  cd $1/HardNet_MultiDataset
#  rm -rf HP_descs/* # run this from time to time
  python -utt HPatches_extract_HardNet.py --model_path="Models/$2/$2.pt" --overwrite --hpatches_dir="$1/hpatches-release"

  cd "$1/hpatches-benchmark/python"
  name=HardNet_$2
  dir="$1/HardNet_MultiDataset/HP_descs"
  python -utt hpatches_eval.py --descr-dir=$dir --descr-name=$name --task=retrieval --split=illum --del=1
  python -utt hpatches_eval.py --descr-dir=$dir --descr-name=$name --task=retrieval --split=view --del=1
  python -utt hpatches_eval.py --descr-dir=$dir --descr-name=$name --task=retrieval --split=full --del=1

  python -utt hpatches_results.py --results-dir=results/ --descr-name=$name --task=retrieval --split=illum
  python -utt hpatches_results.py --results-dir=results/ --descr-name=$name --task=retrieval --split=view
  python -utt hpatches_results.py --results-dir=results/ --descr-name=$name --task=retrieval --split=full
  cd $orig
}

hver(){
  export new_model="/media/milan/Milan_Pultar/No_Backup/Prace/CMP/hardnet/pretrained/$1.pt"
  bbb=$(pwd)
  cd "/media/milan/Milan_Pultar/No_Backup/Prace/CMP/hardnet/benchmarks"
  python HPatches_extract_HardNet.py
  cd "$bbb"
  name="HardNet_$(basename $1)"
  verification HardNet_$1
}

########
# $1 data file
# $2 data column separator
# $3 model_info file
#######
set -u

head -n 1 $1 | awk -F$2 '{$1=$1; print $0;}' OFS='\n' > tmp_feature
awk -F'\t' 'NR>=2{print $1;}' $3 > tmp_head
paste -d"^" tmp_head tmp_feature
rm tmp_head tmp_feature


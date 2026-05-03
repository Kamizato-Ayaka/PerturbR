cat > download_gse92742.sh <<'EOF'
#!/usr/bin/env bash
set -e

mkdir -p GSE92742
cd GSE92742

aria2c -c -x 8 -s 8 -k 4M --file-allocation=none \
  "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE92nnn/GSE92742/suppl/GSE92742%5FBroad%5FLINCS%5Fsig%5Finfo%2Etxt%2Egz"

aria2c -c -x 8 -s 8 -k 4M --file-allocation=none \
  "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE92nnn/GSE92742/suppl/GSE92742%5FBroad%5FLINCS%5Fpert%5Finfo%2Etxt%2Egz"

aria2c -c -x 8 -s 8 -k 4M --file-allocation=none \
  "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE92nnn/GSE92742/suppl/GSE92742%5FBroad%5FLINCS%5Finst%5Finfo%2Etxt%2Egz"

aria2c -c -x 8 -s 8 -k 4M --file-allocation=none \
  "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE92nnn/GSE92742/suppl/GSE92742%5FBroad%5FLINCS%5Fgene%5Finfo%2Etxt%2Egz"

aria2c -c -x 8 -s 8 -k 4M --file-allocation=none \
  "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE92nnn/GSE92742/suppl/GSE92742%5FBroad%5FLINCS%5Fcell%5Finfo%2Etxt%2Egz"

aria2c -c -x 16 -s 16 -k 8M --file-allocation=none \
  --max-tries=0 --retry-wait=10 --timeout=60 \
  "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE92nnn/GSE92742/suppl/GSE92742%5FBroad%5FLINCS%5FLevel3%5FINF%5Fmlr12k%5Fn1319138x12328%2Egctx%2Egz"
EOF

chmod +x download_gse92742.sh
bash download_gse92742.sh
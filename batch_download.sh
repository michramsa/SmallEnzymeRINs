#!/bin/bash

# Script to download files from RCSB http file download services.
# Use the -h switch to get help on usage.

echo "Script started"

if ! command -v curl &> /dev/null
then
    echo "'curl' could not be found. You need to install 'curl' for this script to work."
    exit 1
fi

PROGNAME=$0
BASE_URL="https://files.rcsb.org/download"

usage() {
  cat << EOF >&2
Usage: $PROGNAME -f <file> [-o <dir>] [-c] [-p]

 -f <file>: the input file containing a comma-separated list of PDB ids
 -o  <dir>: the output dir, default: current dir
 -c       : download a cif.gz file for each PDB id
 -p       : download a pdb.gz file for each PDB id (not available for large structures)
 -a       : download a pdb1.gz file (1st bioassembly) for each PDB id (not available for large structures)
 -A       : download an assembly1.cif.gz file (1st bioassembly) for each PDB id
 -x       : download a xml.gz file for each PDB id
 -s       : download a sf.cif.gz file for each PDB id (diffraction only)
 -m       : download a mr.gz file for each PDB id (NMR only)
 -r       : download a mr.str.gz for each PDB id (NMR only)
EOF
  exit 1
}

download() {
  url="$BASE_URL/$1"
  out=$2/$1
  echo "Downloading $url to $out"
  curl -s -f $url -o $out || echo "Failed to download $url"
}

listfile=""
outdir="."
cif=false
pdb=false
pdb1=false
cifassembly1=false
xml=false
sf=false
mr=false
mrstr=false
while getopts f:o:cpaAxsmr o
do
  case $o in
    (f) listfile=$OPTARG;;
    (o) outdir=$OPTARG;;
    (c) cif=true;;
    (p) pdb=true;;
    (a) pdb1=true;;
    (A) cifassembly1=true;;
    (x) xml=true;;
    (s) sf=true;;
    (m) mr=true;;
    (r) mrstr=true;;
    (*) usage
  esac
done
shift "$((OPTIND - 1))"

if [ "$listfile" == "" ]
then
  echo "Parameter -f must be provided"
  exit 1
fi
contents=$(cat $listfile)

# see https://stackoverflow.com/questions/918886/how-do-i-split-a-string-on-a-delimiter-in-bash#tab-top
IFS=',' read -ra tokens <<< "$contents"

echo "Parsed PDB IDs:"
for token in "${tokens[@]}"; do
    echo "$token"
done

for token in "${tokens[@]}"
do
  # Remove "_1" suffix if present
  clean_token="${token%_*}"

  if [ "$cif" == true ]; then
    download "${clean_token}.cif.gz" "$outdir"
  fi
  if [ "$pdb" == true ]; then
    download "${clean_token}.pdb.gz" "$outdir"
  fi
  if [ "$pdb1" == true ]; then
    download "${clean_token}.pdb1.gz" "$outdir"
  fi
  if [ "$cifassembly1" == true ]; then
    download "${clean_token}-assembly1.cif.gz" "$outdir"
  fi
  if [ "$xml" == true ]; then
    download "${clean_token}.xml.gz" "$outdir"
  fi
  if [ "$sf" == true ]; then
    download "${clean_token}-sf.cif.gz" "$outdir"
  fi
  if [ "$mr" == true ]; then
    download "${clean_token}.mr.gz" "$outdir"
  fi
  if [ "$mrstr" == true ]; then
    download "${clean_token}_mr.str.gz" "$outdir"
  fi
done
#!/bin/bash

#this is the main shell script which executes the parameter estimations

EVENT=$6
TYPEWF=$4

if [ "$EVENT" = "170817" ]; then
        t0=1187008882   # GW170817
        wfpath= # your path to the CoRe database waveform for a GW170817-like signal

        if [ "$TYPEWF" = "Easter" ]; then
                results_dir="resultsBaselineEaster170817"
                ITK="itkEaster3_copy"
        else
                #results_dir="results170817"
                results_dir="results_injNRPMw_recEaster_170817"
                ITK="injectToolkit"
        fi

        #INJECT="yes"
        INJECT="no"

elif [ "$EVENT" = "190425" ]; then
        t0=1240215503   # GW190425
        wfpath= # your path

        if [ "$TYPEWF" = "Easter" ]; then
                results_dir="resultsBaselineEaster190425"
                ITK="itkEaster3_copy"
        else
                #results_dir="results190425"
                results_dir="results_injNRPMw_recEaster_190425"
                #ITK="injectToolkit"
                ITK="itkNRPMw_Easter"
        fi

        #INJECT="yes"
        INJECT="no"

elif [ "$EVENT" = "baseline170817" ]; then
        t0=1187008882 #GW170817
        wfpath= # your path
        results_dir="resultsBaseline170817"
        #results_dir="ASDtext170817"
        INJECT="no"
        ITK="injectToolkit"
elif [ "$EVENT" = "baseline190425" ]; then
        t0=1240215503 #GW190425
        wfpath= # your path
        results_dir="resultsBaseline190425"
        #results_dir="ASDtext190425"
        INJECT="no"
        ITK="injectToolkit"
fi
noisepath= # your path to the GWOSC timeseries data

SNR=$1
RANDSEED=$2
TYPENOISE=$3
DISTANCEPERCENT=$5

#Stripping trailing zeroes because 5.0 is worse than 5
SNR=$(printf "%g" "$SNR")
RANDSEED=$(printf "%g" "$RANDSEED")
NAME="${TYPENOISE}_${TYPEWF}"

mins=$((30  * SNR + 300)) # If the allocation time is varaible

DIR="${results_dir}/${TYPENOISE}_${TYPEWF}_SNR${SNR}_SEED${RANDSEED}_${DISTANCEPERCENT}"

mkdir -p "$DIR"
outdir="your_path/$DIR"

cat <<EOF > "$DIR/run_secondary.sh"
#!/bin/bash

#SBATCH --job-name=${NAME}_SNR${SNR}_${RANDSEED}
#SBATCH --partition=milan
#SBATCH --ntasks=2
#SBATCH --time=mins
#SBATCH --mem-per-cpu=500
#SBATCH --chdir=${TYPENOISE}_${TYPEWF}_SNR${SNR}_SEED${RANDSEED}_${DISTANCEPERCENT}

source your_path/pythonrc
python3 your_path/run.py \
--SNR $SNR \
--RANDSEED $RANDSEED \
--t0 $t0 \
--noisepath $noisepath \
--wfpath $wfpath \
--NAME $NAME \
--outdir $outdir \
--typenoise $TYPENOISE \
--typewf $TYPEWF \

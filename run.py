import peToolkit as petk
import os
import argparse

def clean_number(val): # strips trailing zeroes
    if isinstance(val, float) and val.is_integer():
        return str(int(val))
    return str(val)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--RANDSEED", type=int, required=True)
    parser.add_argument("--SNR", type=float, required=True)
    parser.add_argument("--t0", type=float, required=True)
    parser.add_argument("--noisepath", type=str, required=True)
    parser.add_argument("--wfpath", type=str, required=True)
    parser.add_argument("--NAME", type=str, required=True)
    parser.add_argument("--outdir", type=str, required=True)
    parser.add_argument("--typenoise", type=str, required=True)
    parser.add_argument("--typewf", type=str, required=True)
    parser.add_argument("--distancepercent", type=float, required=True)
    parser.add_argument("--inject", type=str, required=False)
    parser.add_argument("--itk", type=str, required=False)

    args = parser.parse_args()

    seed=args.RANDSEED
    SNR=args.SNR
    t_0=args.t0
    noise_path=args.noisepath
    wf_path=args.wfpath
    name=args.NAME
    type_wf=args.typewf
    type_noise=args.typenoise
    distance_percent=args.distancepercent
    inject=args.inject
    itk = args.itk if args.itk is not None else 'injectToolkit'

    seed_strip=clean_number(seed)
    SNR_strip=clean_number(SNR)
    iteration=seed_strip

    outdir=args.outdir
    label=name+'_SNR'+SNR_strip+"_"+seed_strip

    petk.do_pe(SNR,
                seed,
                t_0,
                outdir,
                label,
                noise_path,
                wf_path,
                type_noise,
                type_wf,
                distance_percent,
                inject,
                itk_script=itk
                )

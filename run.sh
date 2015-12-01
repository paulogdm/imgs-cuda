#!/bin/bash

bold=$(tput bold)
normal=$(tput sgr0)
main_mt="main_mt"
N_PROCESS=4
N_EXEC=1

echo ""
while echo $1 | grep -q ^-; do
    declare $( echo $1 | sed 's/^-//' )=$2
    shift
    shift
done

if [[ ! "$bin" ]]; then
	echo "[!] No -bin arg. Found: $bin"
	echo "[?] Searching at bin/"
	bin=( $(ls -d bin/*) )
	echo "[.] Found: ${bin[@]}"
	echo "Done!"
else
	bin=(bin/$bin)
fi


if [[ ! "$fname" ]]; then
	echo "[!] No -fname arg. Found: $fname"
	echo "[?] Searching at in/"
	fname=($(ls -d in/*ppm in/*pgm))
	echo "[.] Found: ${fname[@]}"
	echo "Done!"
else
	fname=(in/$fname)
fi

echo "*************************"
echo "*** Execution Started ***"
echo "[!!!] RUNNING MPI WITH $N_PROCESS PROCESS"
echo ""
for main in ${bin[@]}; do
	for image in ${fname[@]}; do
		echo "Running ${bold} $(basename "$main") ${normal} with ${bold} $(basename "$image") ${normal}"
			for (( i=0; i < N_EXEC; i++ )); do
				if [[ $(basename "$main") == "$main_mt" ]]; then
					mpirun -n $N_PROCESS --bind-to none $main $image ./out/$(basename "$main")_$(basename "$image") >> ./out/$(basename "$main").time
				else
					./$main $image ./out/$(basename "$main")_$(basename "$image") >> ./out/$(basename "$main").time
				fi
		done
		echo "Done!"
	done
done
echo ""
echo "*** Execution Ended ***"
echo "***********************"

echo "[.] Now please, check ${bold} out/ ${normal}"
echo ""

check:
	squeue -u ${USER} -l

clean:
	rm -f *.err *.out
	rm cache*

1:
	sbatch --partition=aquila ./DO.train

2:
	sbatch --partition=aquila ./DO.split

3:
	sbatch --partition=aquila ./DO.generate

# eval on val using the baseline.fix
4:
	sbatch --partition=aquila ./DO.val 

score_val:
	python scripts/scores.py --dataset val --dataroot data/ --outfile pred/val/baseline_val.json --scorefile pred/val/score.txt

tmp:
	srun --pty --jobid  Job_ID /bin/bash

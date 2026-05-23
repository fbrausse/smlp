### 1. Pull the image

```bash
docker pull mdmitry1/python311-dev:latest
```

### 2. Run the container

```bash
docker run -it mdmitry1/python311-dev:latest
```

### 3. In the container

```bash
cd smlp
git checkout remotes/origin/poly_pareto tutorial/examples/si/smlp
xvfb-run tutorial/examples/si/smlp/run_si_test_nosplit
```

### 4. Runtime benchmark results in Docker CLI environment
- Intel(R) Core(TM) i5-7200U CPU @ 2.50GHz 
`  1364.003u 2.921s 22:49.65 99.8% 0+0k 356456+416600io 531pf+0w`
- Mac M1 Docker CLI environment
`   653.405u 1.463s 10:53.50 100.2% 0+0k 5736+417504io 26pf+0w`

### 6. Expected results plot
<img src="media/no_split_s2_tx_piv_anonym_optimization_results.png" width="75%"/>

### 7. Numerical diffs validation
```bash
diff no_split_s2_tx_piv_anonym_optimization_results_relative_optimized_margin.txt results/NO_SPLIT_S2_TX_PIV_ANONYM_OPTIMIZATION_RESULTS_RELATIVE_OPTIMIZED_MARGIN_EXPECTED.txt
```

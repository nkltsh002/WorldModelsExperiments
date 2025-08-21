mkdir -p results
docker build -t world-models-test-volume -f Dockerfile.test_volume .
docker run --rm -v ${PWD}/results:/app/results world-models-test-volume
echo "Test results:"
cat results/test_results.txt

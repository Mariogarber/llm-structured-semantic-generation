kubectl apply -f labeled_code.yaml
kubectl wait --for=condition=complete jobs/example --timeout=60s

if [ $(kubectl logs --selector job-name=example | grep "OK" | wc -l) -ne 10 ]; then exit 1; fi
sleep 10
if [ $(kubectl logs --selector job-name=example | grep "OK" | wc -l) -eq 0 ]; then echo "cloudeval_unit_test_passed_2"; fi
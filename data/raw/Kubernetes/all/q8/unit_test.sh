kubectl apply -f labeled_code.yaml
sleep 70
if [[ $(kubectl get jobs | wc -l) -eq 0 ]] && [[ $(kubectl get cronjobs | grep '0 0 1 \* \*') ]]; then
    echo "cloudeval_unit_test_passed"
fi
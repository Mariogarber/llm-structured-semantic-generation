kubectl apply -f labeled_code.yaml
kubectl wait --for=condition=Ready pod -l name=ds-grace-serviceaccount --timeout=60s

pods=$(kubectl get pods -l name=ds-grace-serviceaccount --output=jsonpath={.items..metadata.name})

grace_period=$(kubectl get pod $pods -o=jsonpath='{.spec.terminationGracePeriodSeconds}')
if [[ "$grace_period" -ne 60 ]]; then
    exit 1
fi

service_account=$(kubectl get pod $pods -o=jsonpath='{.spec.serviceAccountName}')
if [[ "$service_account" != "default" ]]; then
    exit 1
fi

echo cloudeval_unit_test_passed

kubectl apply -f labeled_code.yaml

sleep 10
if [[ $(kubectl get pv gce-pv -o=jsonpath='{.status.phase}') != "Available" ]]; then
    kubectl delete -f labeled_code.yaml
    exit 1
fi

pv_description=$(kubectl describe pv gce-pv)

if echo "$pv_description" | grep "PDName" && echo "$pv_description" | grep "FSType" && echo "$pv_description" | grep "GCEPersistentDisk"; then
  echo cloudeval_unit_test_passed
else
  echo "Test failed"
fi
kubectl delete -f labeled_code.yaml
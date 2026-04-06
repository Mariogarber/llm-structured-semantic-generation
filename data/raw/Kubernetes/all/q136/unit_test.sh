kubectl apply -f labeled_code.yaml

sleep 10
if [[ $(kubectl get pv glusterfs-pv -o=jsonpath='{.status.phase}') != "Available" ]]; then
    kubectl delete -f labeled_code.yaml
    exit 1
fi
pv_description=$(kubectl describe pv glusterfs-pv)

if echo "$pv_description" | grep "\s*EndpointsName:\s*glusterfs-cluster" && echo "$pv_description" | grep "Type:\s*Glusterfs"; then
  echo cloudeval_unit_test_passed
else
  echo "Test failed"
fi
kubectl delete -f labeled_code.yaml
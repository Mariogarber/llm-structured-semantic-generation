kubectl apply -f labeled_code.yaml

sleep 20
if [[ $(kubectl get pv nfs-pv -o=jsonpath='{.status.phase}') != "Available" ]]; then
    kubectl delete -f labeled_code.yaml
    exit 1
fi

pv_description=$(kubectl describe pv nfs-pv)

# Check for the NFS server and path details
if echo "$pv_description" | grep -q "Server:\s*10.108.211.55" && echo "$pv_description" | grep -q "Path:\s*/var/nfs"; then
  echo cloudeval_unit_test_passed
else
  echo "Test failed"
fi
kubectl delete -f labeled_code.yaml

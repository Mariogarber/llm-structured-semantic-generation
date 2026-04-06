kubectl apply -f labeled_code.yaml

sleep 10
if [[ $(kubectl get pv iscsi-pv -o=jsonpath='{.status.phase}') != "Available" ]]; then
    kubectl delete -f labeled_code.yaml
    exit 1
fi

desc=$(kubectl describe pv iscsi-pv)

if echo "$desc" | grep "TargetPortal" | grep "10.0.0.1:3260"  && echo "$desc" | grep "Lun" | grep "0" && echo "$desc" | grep "FSType" | grep "ext4"; then
  echo cloudeval_unit_test_passed
else
  echo "Test failed"
fi
kubectl delete -f labeled_code.yaml
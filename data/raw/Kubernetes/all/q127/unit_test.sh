cat <<EOF | kubectl apply -f -
kind: PersistentVolume
apiVersion: v1
metadata:
  name: mysql-pv-volume
  labels:
    type: local
spec:
  storageClassName: manual
  capacity:
    storage: 15Gi
  accessModes:
    - ReadWriteOnce
  hostPath:
    path: "/mnt/data"
EOF

cleanup() {
    kubectl delete -f labeled_code.yaml
    kubectl delete pv mysql-pv-volume
}

sleep 10
if [[ $(kubectl get pv mysql-pv-volume -o=jsonpath='{.status.phase}') != "Available" ]]; then
    cleanup
    exit 1
fi
kubectl apply -f labeled_code.yaml

sleep 5
# Check the bound PV for the PVC
bound_pv=$(kubectl get pvc mysql-pv-claim -o=jsonpath='{.spec.volumeName}')
echo $bound_pv
# Check if the PVC is bound to the correct PV
if [ "$bound_pv" == "mysql-pv-volume" ]; then
    echo "PVC is bound to the correct PV."
else
    cleanup
    exit 1
fi

# Check the details of the PV
desc=$(kubectl describe pv "$bound_pv")
if [[ $desc == *"/mnt/data"* ]] && 
   [[ $desc == *"StorageClass:    manual"* ]]; then
    echo cloudeval_unit_test_passed
else
    echo "Test failed"
fi
cleanup


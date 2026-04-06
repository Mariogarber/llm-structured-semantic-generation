kubectl apply -f labeled_code.yaml
sleep 10
if [[ $(kubectl get pv mysql-pv-volume -o=jsonpath='{.status.phase}') != "Available" ]]; then
    kubectl delete -f labeled_code.yaml
    exit 1
fi

cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: mysql-pv-claim
spec:
  storageClassName: manual
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 10Gi
EOF

cleanup() {
    kubectl delete pvc mysql-pv-claim
    kubectl delete -f labeled_code.yaml
}

# Wait for PVC to get bound
sleep 5
pvc_status=$(kubectl get pvc mysql-pv-claim -o=jsonpath='{.status}')
storage_class=$(kubectl get pvc mysql-pv-claim -o=jsonpath='{.spec.storageClassName}')

if echo "$pvc_status" | grep Bound && echo "$storage_class" | grep manual; then
    echo "PVC passed"
else
    cleanup
    exit 1
fi

# Check the details of the PV
storage_value=$(kubectl describe pv mysql-pv-volume | awk '/StorageClass:/ {print $2}')
capacity_value=$(kubectl describe pv mysql-pv-volume | awk '/Capacity:/ {print $2}')
modes_value=$(kubectl describe pv mysql-pv-volume | awk '/Access Modes:/ {print $3}')
host_value=$(kubectl describe pv mysql-pv-volume | awk '/Path:/ {print $2}')
if [ "$storage_value" == "manual" ] && \
   [ "$capacity_value" == "15Gi" ] && \
   [ "$modes_value" == "RWO" ] && \
   [ "$host_value" == "/mnt/data" ]; then
  echo cloudeval_unit_test_passed
else
  echo "Test failed"
fi
cleanup
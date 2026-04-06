kubectl label nodes minikube topology.kubernetes.io/zone=us-central1-a
kubectl apply -f labeled_code.yaml
sleep 10
if [[ $(kubectl get pv test-volume -o=jsonpath='{.status.phase}') != "Available" ]]; then
    kubectl label nodes minikube topology.kubernetes.io/zone-
    kubectl delete -f labeled_code.yaml
    exit 1
fi
# Create a PVC to claim the PV
cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: test-pvc
spec:
  accessModes:
    - ReadWriteOnce
  storageClassName: storage-test-volume
  resources:
    requests:
      storage: 4Gi
EOF

cleanup() {
    kubectl label nodes minikube topology.kubernetes.io/zone-
    kubectl delete pvc test-pvc
    kubectl delete -f labeled_code.yaml
}

# Wait for PVC to get bound
sleep 5
pvc_status=$(kubectl get pvc test-pvc -o=jsonpath='{.status}')
storage_class=$(kubectl get pvc test-pvc -o=jsonpath='{.spec.storageClassName}')

if echo "$pvc_status" | grep Bound && echo "$storage_class" | grep storage-test-volume; then
    echo cloudeval_unit_test_passed
else
    cleanup
    exit 1
fi

cleanup

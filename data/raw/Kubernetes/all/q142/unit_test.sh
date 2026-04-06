kubectl apply -f labeled_code.yaml

sleep 10
if [[ $(kubectl get pv pv0001 -o=jsonpath='{.status.phase}') != "Available" ]]; then
    kubectl delete -f labeled_code.yaml
    exit 1
fi

cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: test-pvc
spec:
  accessModes:
    - ReadWriteOnce
  storageClassName: slow
  resources:
    requests:
      storage: 4Gi
EOF

cleanup() {
    kubectl delete pvc test-pvc
    kubectl delete -f labeled_code.yaml
}

sleep 5
pvc_status=$(kubectl get pvc test-pvc -o=jsonpath='{.status}')
storage_class=$(kubectl get pvc test-pvc -o=jsonpath='{.spec.storageClassName}')

if echo "$pvc_status" | grep Bound && echo "$storage_class" | grep slow; then
    echo "a"
else
    cleanup
    exit 1
fi

echo $(kubectl describe pv pv0001)
type_value=$(kubectl describe pv pv0001 | awk '/Type:/ {print $2}')
echo $type_value
server_value=$(kubectl describe pv pv0001 | awk '/Server:/ {print $2}')
echo $server_value
path_value=$(kubectl describe pv pv0001 | awk '/Path:/ {print $2}')
echo $path_value
readonly_value=$(kubectl describe pv pv0001 | awk '/ReadOnly:/ {print $2}')
echo $readonly_value
if [ "$type_value" == "NFS" ] && \
   [ "$server_value" == "172.17.0.2" ] && \
   [ "$path_value" == "/tmp" ] && \
   [ "$readonly_value" == "false" ]; then
  echo cloudeval_unit_test_passed
else
  echo "Test failed"
fi

cleanup
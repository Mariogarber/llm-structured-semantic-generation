kubectl apply -f labeled_code.yaml

cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: valid-pvc
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 1.5Gi
EOF

sleep 5
# Create a PVC that exceeds the LimitRange and capture any errors
ERROR_MESSAGE=$(cat <<EOF | kubectl apply -f - 2>&1
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: exceed-pvc
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 2.5Gi
EOF
)

if [[ $ERROR_MESSAGE == *"PersistentVolumeClaim is 2Gi"* ]]; then
    echo cloudeval_unit_test_passed
else
    kubectl delete pvc valid-pvc
    echo "Test failed"
fi




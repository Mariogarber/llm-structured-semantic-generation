kubectl apply -f labeled_code.yaml

cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: hostpath-pvc
spec:
  storageClassName: manual
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 1Gi
EOF

cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: Pod
metadata:
  name: test-hostpath-pv
spec:
  volumes:
    - name: test-volume
      persistentVolumeClaim:
        claimName: hostpath-pvc
  containers:
    - name: test-container
      image: busybox
      volumeMounts:
        - mountPath: "/mnt/test"
          name: test-volume
      command: ["/bin/sh"]
      args: ["-c", "touch /mnt/test/hello_from_pv; sleep 3600"]
EOF

sleep 10

if kubectl exec test-hostpath-pv -- ls /mnt/test/hello_from_pv; then
  echo cloudeval_unit_test_passed
else
  echo "Test failed"
fi

kubectl delete pod test-hostpath-pv
kubectl delete pvc hostpath-pvc
kubectl delete -f labeled_code.yaml

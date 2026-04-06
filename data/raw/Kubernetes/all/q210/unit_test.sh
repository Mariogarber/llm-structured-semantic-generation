kubectl apply -f labeled_code.yaml

cat <<EOF | kubectl apply -f -
apiVersion: scheduling.k8s.io/v1
kind: PriorityClass
metadata:
  name: cluster-services
value: 1000
EOF

cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: PersistentVolume
metadata:
  name: my-volume
spec:
  capacity:
    storage: 1Gi
  volumeMode: Filesystem
  accessModes:
    - ReadWriteOnce
  storageClassName: standard
  hostPath:
    path: /data/my-volume
EOF

cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: test-pvc
spec:
  storageClassName: standard
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 1Gi
EOF

sleep 5

cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: Pod
metadata:
  name: test-pod
spec:
  priorityClassName: cluster-services
  containers:
    - name: test-container
      image: busybox
      command: ["/bin/sh"]
      args: ["-c", "echo 'Test Pod running'; sleep 3600"]
      volumeMounts:
        - name: test-volume
          mountPath: /mnt/data
  volumes:
    - name: test-volume
      persistentVolumeClaim:
        claimName: test-pvc
EOF

sleep 5

priority_class=$(kubectl describe pod test-pod | grep "Priority Class")

if [[ $(kubectl get resourcequota | grep "pods-cluster-services") ]] && [[ $priority_class == *"cluster-services"* ]]; then
    echo "cloudeval_unit_test_passed"
else
    echo "Test failed"
fi

kubectl delete pod test-pod
kubectl delete pvc test-pvc
kubectl delete pv my-volume
kubectl delete priorityclass cluster-services
kubectl delete -f labeled_code.yaml
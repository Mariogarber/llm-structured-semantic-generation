echo "apiVersion: v1
kind: PersistentVolume
metadata:
  name: mysql-pv-volume
  labels:
    type: local
spec:
  storageClassName: manual
  capacity:
    storage: 1Gi
  accessModes:
    - ReadWriteOnce
  hostPath:
    path: "/mnt/data"
---
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
      storage: 1Gi
" | kubectl apply -f -

kubectl wait pv --all --for=jsonpath='{.status.phase}'=Bound --timeout=20s
kubectl wait pvc --all --for=jsonpath='{.status.phase}'=Bound --timeout=20s

kubectl apply -f labeled_code.yaml

kubectl wait deployments --all --for=condition=available --timeout=40s

# make sure:
# 1/1 deployments are Ready
kubectl get deployments | awk -v RS='' '\
/1\/1/ \
{print "cloudeval_unit_test_passed"}'
# timeout 20 kubectl run -it --rm --image=mysql:latest --restart=Never mysql-client -- mysql -h mysql -ppassword | awk -v RS='' '\
# /enter/ || \
# /mysql>/ \
# {print "cloudeval_unit_test_passed"}'
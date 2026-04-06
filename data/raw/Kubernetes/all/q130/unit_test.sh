kubectl apply -f labeled_code.yaml
kubectl describe persistentvolume pv-name | grep "Type:      NFS (an NFS mount that lasts the lifetime of a pod)
    Server:    255.0.255.0
    Path:      /desired/path/in/nfs" && echo cloudeval_unit_test_passed
# Stackoverflow https://stackoverflow.com/questions/31693529/how-to-share-storage-between-kubernetes-pods
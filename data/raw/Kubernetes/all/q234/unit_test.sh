kubectl apply -f labeled_code.yaml
sleep 3
kubectl describe secret mysql-secret | grep "mysql-root-password:  6 bytes
mysql-user:           2 bytes
mysql-password:       6 bytes" && echo cloudeval_unit_test_passed
# Stackoverflow: https://stackoverflow.com/questions/75129018/error-from-server-badrequest-error-when-creating-stdin-secret-in-version
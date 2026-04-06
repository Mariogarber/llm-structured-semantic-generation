kubectl apply -f labeled_code.yaml

[ "$(kubectl get deployment wordpress-mysql -o jsonpath='{.spec.template.spec.containers[0].env[0].valueFrom.secretKeyRef.name}')" = "mysql-pass" ] && \
[ "$(kubectl get deployment wordpress-mysql -o jsonpath='{.spec.template.spec.containers[0].env[0].valueFrom.secretKeyRef.key}')" = "password" ] && \
[ "$(kubectl get deployment wordpress-mysql -o jsonpath='{.spec.template.spec.containers[0].env[?(@.name=="MYSQL_DATABASE")].value}')" = "wordpress" ] && \
[ "$(kubectl get deployment wordpress-mysql -o jsonpath='{.spec.template.spec.containers[0].env[?(@.name=="MYSQL_USER")].value}')" = "wordpress" ] && \
[ "$(kubectl get secret mysql-pass -o jsonpath='{.data.password}' | base64 --decode)" = "mypassword" ] && \
echo cloudeval_unit_test_passed

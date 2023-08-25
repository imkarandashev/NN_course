<div class="tl_page_wrap">

<div class="tl_page">

<main class="tl_article">

<header class="tl_article_header" dir="auto">

# Torch grad

<address><a rel="author"></a><time datetime="2020-12-26T18:56:02+0000">December 26, 2020</time></address>

</header>

<article id="_tl_editor" class="tl_article_content">

# Torch grad  

<address>  
</address>

Автоматическое дифференцирование

Механизм автоматического дифференцирования находится в модуле torch.autograd. С помощью него можно автоматически считать производны, что необходимо для многих задач оптимизации, в том числе для обучения нейронных сетей.

Все тензоры содержат в себе атрибут .requires_grad. Если обозначить его как `True`, то все операции над тензорами начнут сохраняться в граф вычислений, и при вызове `.backward()` выполнит обратный проход и посчитает производную по всем переменным, при создании которых мы указали `requires_grad=True`. Производная при этом запишется в свойство `x.grad`. Получить же свой тензор обратно мы можем используя свойство `x.data`. Свойство `x.requires_grad` покажет, нуждается ли узел графа в вычислении градиента. Правило такое: если хоть у одного дочернего узла это свойство установлено, оно будет установлено и у родителя.

Если обернуть тензоры в `Variable` то он сам позаботится о сохранении всех методов и свойств, и будет записывать все операции.

Что бы тензор не отслеживал историю, вы можете вызвать метод `.detach()`, что бы отделить историю вычислений от тензора. Так же это остановит дальнейшее отслеживание вычислений.

Что бы предотвратить отслеживание истории и использование памяти, вы можете обернуть блок кода в `with torch.no_grad():`. Это может быть полезно при оценке модели, так как модель может иметь обучающиеся параметры с requires_grad=True, но для которых не нужно считать градиент.

`Tensor` и `Function` взаимосвязаны и строятся как ациклический граф, который кодирует историю вычислений. Каждый тензор имеет атрибут `.grad_fn` который ссылается на функцию, которая создала тензор(исключение тензор который создал пользователь, тогда `grad_fn is None`).

Поскольку при вызове `backward()` устанавливается член `grad` у `Variables`, также существует метод `nn.Module.zero_grad()`, он обнуляет всю историю подсчёта градиентов. Обычно этот метод вызывается перед вызовом `backward()`, чтобы можно было провести следующий шаг оптимизации.

На данный момент autograd поддерживается только для тензоров с плавающей запятой (half, float, double и bfloat16) и сложных тензоров типов (cfloat, cdouble).

Пример:

<pre>import torch
from torch.autograd import Variable
T = torch.Tensor([[1., 2.], [2., 3.]])
T = Variable(T, requires_grad=True)
print(T.requires_grad) 
True

T1 = torch.sqrt(T)
print(T1)  
tensor([[1.0000, 1.4142],
        [1.4142, 1.7321]], grad_fn=<SqrtBackward>)

T2 = T1.sum()
print(T2)
tensor(5.5605, grad_fn=<SumBackward0>)

T2.backward()
print(T.grad)
tensor([[1.5000, 1.3536],
        [1.3536, 1.2887]])

</pre>

Разберём пример подробнее:

После импорта библиотек инициализируем тензор T, оборачиваем его в Variable и обозначаем переменную requires_grad=True для записи графа действий.

T = torch.Tensor([[1., 2.], [2., 3.]])

T = Variable(T, requires_grad=True)

print(T.requires_grad)

True

Проводим операцию взятия корня. Как можно видеть теперь в тензоре хранится последнее действие с ним, в данном случае взятие корня (grad_fn=<SqrtBackward>).

<pre>T1 = torch.sqrt(T)
print(T1)  
tensor([[1.0000, 1.4142],
        [1.4142, 1.7321]], grad_fn=<SqrtBackward>)
</pre>

Так как .backward() работает только со скалярными значениями, то проводим операцию суммы. Как можно видеть тензор сохранил эту операцию

<pre>T2 = T1.sum()
print(T2)
tensor(5.5605, grad_fn=<SumBackward0>)
</pre>

Вычисляем градиенты с помощью .backward() и выводим на экран посчитанные градиенты для T.

<pre>T2.backward()
print(T.grad)
tensor([[1.5000, 1.3536],
        [1.3536, 1.2887]])
</pre>

</article>

<div id="_tl_tooltip" class="tl_tooltip">

<div class="buttons"><span class="button_hover"></span><span class="button_group"></span><span class="button_group"></span></div>

<div class="prompt"><span class="close"></span>

<div class="prompt_input_wrap"><input type="url" class="prompt_input"></div>

</div>

</div>

<aside class="tl_article_buttons"><button id="_edit_button" class="button edit_button">Edit</button><button id="_publish_button" class="button publish_button">Publish</button></aside>

</main>

</div>

<div class="tl_page_footer">

<div id="_report_button" class="tl_footer_button">Report content on this page</div>

</div>

</div>

<div class="tl_popup tl_popup_hidden" id="_report_popup">

<main class="tl_popup_body tl_report_popup">

<form id="_report_form" method="post">

<section>

## Report Page

<div class="tl_radio_items"><label class="tl_radio_item"><input type="radio" class="radio" name="reason" value="violence"> <span class="tl_radio_item_label">Violence</span> </label> <label class="tl_radio_item"> <input type="radio" class="radio" name="reason" value="childabuse"> <span class="tl_radio_item_label">Child Abuse</span> </label> <label class="tl_radio_item"> <input type="radio" class="radio" name="reason" value="copyright"> <span class="tl_radio_item_label">Copyright</span> </label> <label class="tl_radio_item"> <input type="radio" class="radio" name="reason" value="illegal_drugs"> <span class="tl_radio_item_label">Illegal Drugs</span> </label> <label class="tl_radio_item"> <input type="radio" class="radio" name="reason" value="personal_details"> <span class="tl_radio_item_label">Personal Details</span> </label> <label class="tl_radio_item"><input type="radio" class="radio" name="reason" value="other"> <span class="tl_radio_item_label">Other</span></label> </div>

<div class="tl_textfield_item tl_comment_field"><input type="text" class="tl_textfield" name="comment" value="" placeholder="Add Comment…"></div>

<div class="tl_copyright_field">Please submit your DMCA takedown request to [dmca@telegram.org](mailto:dmca@telegram.org?subject=Report%20to%20Telegraph%20page%20%22Torch%20grad%22&body=Reported%20page%3A%20https%3A%2F%2Ftelegra.ph%2FTorch-grad-12-26%0A%0A%0A)</div>

</section>

<aside class="tl_popup_buttons"><button type="reset" class="button" id="_report_cancel">Cancel</button> <button type="submit" class="button submit_button">Report</button></aside>

</form>

</main>

</div>

<script>var T={"apiUrl":"https:\/\/edit.telegra.ph","datetime":1609008962,"pageId":"acb75d5e82c53622b8891","editable":true};(function(){var b=document.querySelector('time');if(b&&T.datetime){var a=new Date(1E3*T.datetime),d='January February March April May June July August September October November December'.split(' ')[a.getMonth()],c=a.getDate();b.innerText=d+' '+(10>c?'0':'')+c+', '+a.getFullYear()}})();</script>
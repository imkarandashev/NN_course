<div class="tl_page_wrap">

<div class="tl_page">

<main class="tl_article">

<header class="tl_article_header" dir="auto">

# Datalouder

<address><a rel="author"></a><time datetime="2021-02-11T18:07:25+0000">February 11, 2021</time></address>

</header>

<article id="_tl_editor" class="tl_article_content">

# Datalouder  

<address>  
</address>

Для удобства в PyTorch предоставляются ряд утилит для загрузки датасетов, их предварительной обработки и взаимодействия с ними. Эти вспомогательные классы находятся в модуле `torch.utils.data`. Здесь следует обратить внимание на две основные концепции:

1.  `Dataset`, инкапсулирующий источник данных,
2.  `DataLoader`, отвечающий за загрузку датасета, возможно, в параллельном режиме.

`torch.utils.data.Dataset`- абстрактный класс, представляющий набор данных.

Самый важный аргумент `DataLoader` конструктора - `dataset` это объект набора данных, из которого загружаются данные. PyTorch поддерживает два разных типа наборов данных:

*   наборы данных в стиле карты ,
*   наборы данных в итерационном стиле .

### Наборы данных в стиле карты

Набор данных в стиле карты является тот , который реализует `__getitem__()` и `__len__()` протоколы, и представляет собой карту из (возможно нецелых) индексов/ключей к выборкам данных.

Например, такой набор данных при доступе с помощью `dataset[idx]`мог бы прочитать `idx`-е изображение и соответствующую ему метку из папки на диске.

### Наборы данных в итерационном стиле

Набор данных итеративного стиля является экземпляром подкласса `IterableDataset` , реализующего `__iter__()` протокол, и представляет собой итерацию по выборкам данных. Этот тип наборов данных особенно подходит для случаев, когда случайное чтение дорого или даже маловероятно, и где размер пакета зависит от извлеченных данных.

Например, такой набор данных при вызове `iter(dataset)`может возвращать поток считанных данных из базы данных, удаленного сервера или даже журналов, созданных в реальном времени.

`DataLoader`поддерживает автоматическое соединение отдельных выборок данных в пакеты с помощью аргументов `batch_size`, `drop_last` и `batch_sampler`.

### Переопределение классов

`Dataset`

Ваш собственный набор данных должен наследовать `Dataset`и переопределять следующие методы:

*   Внутри `__init__` обычно конфигурируются какие-либо пути или изменяется набор возвращаемых в конечном итоге образцов.
*   `__len__` так чтобы `len(dataset)` возвращал размер набора данных. указывается верхний предел индекса, с которым может быть вызван `__getitem__`
*   `__getitem__` для поддержки индексации, в которой `dataset[i]` может использоваться для получения конкретного элемента

DataLoader

Чтобы перебрать датасет, можно, в принципе, применить цикл `for i in range` и обращаться к образцам при помощи `__getitem__`. Однако, было бы гораздо удобнее, если бы датасет сам реализовывал протокол итератора, и мы могли бы сами перебирать образцы при помощи `for sample in dataset`. К счастью, такой функционал предоставляется в классе `DataLoader`. Объект `DataLoader` принимает датасет и ряд опций, конфигурирующих процедуру извлечения образца. Например, можно параллельно загружать образцы, задействовав множество процессов. Для этого конструктор `DataLoader` принимает аргумент `num_workers`. Обратите внимание: `DataLoader` всегда возвращает пакеты, размер которых задается в параметре `batch_size`.

`DataLoader` содержит довольно нетривиальную логику, определяющую, как _комплектовать_ отдельные образцы, возвращенные в методе `__getitem__` вашего датасета в очередной пакет, возвращаемый `DataLoader` при переборе. Например, если `__getitem__` возвращает словарь, то `DataLoader` агрегирует значения этого словаря в единое отображение, соответствующее одному пакету, использующему одинаковые ключи. Это значит, что, если метод `__getitem__` датасета возвращает `dict(example=example, label=label)`, то пакет, возвращенный `DataLoader`, вернет нечто наподобие `dict(example=[example1, example2, ...], label=[label1, label2, ...])`, то есть, распаковывая значения отдельных образцов, мы переупаковываем их под единым ключом для словаря пакета. Чтобы переопределить это поведение, можно передать аргумент функции для параметра `collate_fn` объекту `DataLoader`.

CLASS

`torch.utils.data.DataLoader`(_dataset: torch.utils.data.dataset.Dataset[T_co], batch_size: Optional[int] = 1, shuffle: bool = False, sampler: Optional[torch.utils.data.sampler.Sampler[int]] = None, batch_sampler: Optional[torch.utils.data.sampler.Sampler[Sequence[int]]] = None, num_workers: int = 0, collate_fn: Callable[List[T], Any] = None, pin_memory: bool = False, drop_last: bool = False, timeout: float = 0, worker_init_fn: Callable[int, None] = None, multiprocessing_context=None, generator=None, *, prefetch_factor: int = 2, persistent_workers: bool = False_)

*   dataset( _Dataset_ ) - набор данных, из которого нужно загрузить данные.
*   batch_size ( _int , необязательно_ ) - сколько образцов в партии загружать (по умолчанию : 1) .
*   shuffle ( _bool , необязательно_ ) - установите, `True` чтобы данные перетасовывались в каждую эпоху (по умолчанию:False).
*   sampler ( _Sampler или Iterable , необязательно_ ) - определяет стратегию извлечения выборок из набора данных. Может быть любой `Iterable` с реализованным `__len__` . Если указано, `shuffle` указывать нельзя.
*   batch_sampler ( _Sampler или Iterable , необязательно_ ) - аналогично `sampler`, но возвращает несколько индексов за раз. Взаимоисключающий с `batch_size`, `shuffle`, `sampler`, и `drop_last`.
*   num_workers ( _int , необязательно_ ) - сколько параллельных процессоров использовать для загрузки данных. `0` означает, что данные будут загружены в основной процесс. ( по умолчанию: `0`)
*   collate_fn ( _вызываемая , опционально_ ) - объединяет список образцов, чтобы сформировать мини-партию тензора(ов). Используется при пакетной загрузке из набора данных в стиле карты.
*   pin_memory ( _bool , необязательно_ ) - Если True, загрузчик данных скопирует тензоры в закрепленную память CUDA перед их возвратом.
*   drop_last ( _bool , необязательно_ ) - установите значение True, чтобы отбросить последний неполный пакет, если размер набора данных не делится на размер пакета. Если False и размер набора данных не делится на размер пакета, то последний пакет будет меньше. (по умолчанию: False)
*   timeout ( _numeric , optional_ ) - если положительный, значение тайм-аута для сбора пакета от воркеров. Всегда должно быть неотрицательным. (по умолчанию: `0`)
*   worker_init_fn ( _вызываемый , необязательный_ ) - если не `None`, то это будет вызвано для каждого процесса с его id (int в `[0, num_workers - 1]`) для каждого входа после заполнения и перед загрузкой данных
*   prefetch_factor ( _int , optional , keyword-only arg_ ) - количество выборок, загруженных заранее каждым исполнителем. `2` означает, что будет предварительно загружено 2 * num_workers семпла для всех воркеров. (по умолчанию: `2`)
*   persistent_workers ( _bool , необязательно_ ) - если `True` загрузчик данных не завершит работу рабочих процессов после того, как набор данных был использован один раз. Это позволяет поддерживать рабочие экземпляры Dataset. ( по умолчанию: `False`)

### Автоматический батчинг (по умолчанию)

Это наиболее распространенный случай, который соответствует выборке минибатча и объединению их в пакетные выборки, т.е. содержащие тензоры с измерением, являющимся измерением батча (обычно первым).

Если `batch_size`(по умолчанию `1`) не `None`, то загрузчик данных выдает пакетные выборки вместо отдельных выборок. `batch_size` и `drop_last` аргументы используются, чтобы указать, как загрузчик данных получает пакеты ключей набора данных. Для наборов данных в стиле карты пользователи могут альтернативно указать `batch_sampler`, что дает список ключей за раз.

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

<div class="tl_copyright_field">Please submit your DMCA takedown request to [dmca@telegram.org](mailto:dmca@telegram.org?subject=Report%20to%20Telegraph%20page%20%22Datalouder%22&body=Reported%20page%3A%20https%3A%2F%2Ftelegra.ph%2FDatalouder-02-11%0A%0A%0A)</div>

</section>

<aside class="tl_popup_buttons"><button type="reset" class="button" id="_report_cancel">Cancel</button> <button type="submit" class="button submit_button">Report</button></aside>

</form>

</main>

</div>

<script>var T={"apiUrl":"https:\/\/edit.telegra.ph","datetime":1613066845,"pageId":"5cccb1f1efa90b63605c8","editable":true};(function(){var b=document.querySelector('time');if(b&&T.datetime){var a=new Date(1E3*T.datetime),d='January February March April May June July August September October November December'.split(' ')[a.getMonth()],c=a.getDate();b.innerText=d+' '+(10>c?'0':'')+c+', '+a.getFullYear()}})();</script>